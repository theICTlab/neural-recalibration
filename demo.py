import gradio as gr
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
import sys, os, math, time
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation
import argparse
import plotly.graph_objects as go

from utils import *  # provides expand_parameters, etc.
from diffproj import *  # provides compute_geodesic_distance_from_two_matrices, project_points
from camera_pose_synthesis import *  # provides CameraPoseSynthesizer
import networks
import draw_utilities

torch.cuda.empty_cache()
torch.manual_seed(42)
number_of_devices = torch.cuda.device_count()
print(f'Number of GPU devices: {number_of_devices}')
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
print(f'Using {device}')

# --- Constants ---
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1600
EPOCHS = 20000
WARMUP_EPOCHS = 100 
TEST_SIZE = 20000
STARTING_EPOCH = 0
NUMBER_OF_CAMERAS = 10  # used in original
NUMBER_OF_FIDUCIALS = None
INTRINSIC_VARIATION = 0.05
EXTRINSIC_VARIATION = 0.05
epsilon = 1e-5
LEARNING_RATES = [0.0001]*6
POWER = 1
SMALL_VALUE_SCALE = 1000
BATCH_SIZE = 512

# --- Minimal helper ---
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
timestamp_str = get_timestamp()

def text_to_figure(text):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=20))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

stop_training = False
def stop_training_fn():
    global stop_training
    stop_training = True
    return "Stop training requested."

# --- Override draw_points to convert hex colors ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
def draw_points_fixed(fig, points, colors):
    import itertools
    cyclic_colors = itertools.cycle(colors)
    colors_for_points = [next(cyclic_colors) for _ in range(len(points))]
    rgba_colors = []
    for color in colors_for_points:
        if isinstance(color, str):
            r, g, b = hex_to_rgb(color)
        else:
            r, g, b = color
        rgba_colors.append(f'rgba({r}, {g}, {b}, 1.0)')
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                               mode='markers',
                               marker=dict(size=4, color=rgba_colors, opacity=0.75),
                               showlegend=False))
draw_utilities.draw_points = draw_points_fixed

# --- Network definitions ---
class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, nhead=4,
                 num_encoder_layers=4, num_output_transformer_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.output_transformer = nn.TransformerEncoder(layer, num_layers=num_output_transformer_layers)
        self.final_linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.embedding(x)
        x = self.output_transformer(x)
        return self.final_linear(x)

class DGCCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(NUMBER_OF_FIDUCIALS*2, 512)  # relies on global NUMBER_OF_FIDUCIALS
        layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=8)
        self.R = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=6,
                                   num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.t = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=3,
                                   num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.fc = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=2,
                                    num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.pp = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=2,
                                    num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.kc = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=5,
                                    num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
    def camera_identity_encoding(self, num_cameras, d_model):
        identity = torch.zeros(num_cameras, d_model, device=device)
        for i in range(num_cameras):
            identity[i, i % d_model] = 1
        identity += torch.randn(num_cameras, d_model, device=device) * 0.01
        return identity
    def forward(self, x_in):
        # x_in assumed shape: [B, Nc, Nf, 2]
        batch_size, num_cameras, _, _ = x_in.shape
        x_in_flat = x_in.view(-1, NUMBER_OF_FIDUCIALS*2)
        x_enc = self.embedding(x_in_flat)
        identity_enc = self.camera_identity_encoding(num_cameras, 512).unsqueeze(0).repeat(batch_size,1,1)
        x_enc = x_enc.view(batch_size, num_cameras, -1) + identity_enc
        x = self.encoder(x_enc)
        x_flat = x.view(batch_size*num_cameras, -1)
        x_R = self.R(x_flat)
        x_t = self.t(x_flat)
        x_fc = self.fc(x_flat)
        x_pp = self.pp(x_flat)
        x_kc = self.kc(x_flat)
        x_out = torch.cat([x_R, x_t, x_fc, x_pp, x_kc], dim=-1)
        return x_out.view(batch_size, num_cameras, -1)

# --- Loss functions (same as original) ---
def rmse_loss_fn(predicted, target):
    mse = F.mse_loss(predicted, target, reduction='mean')
    return torch.sqrt(mse)
def logcosh_loss_fn(predicted, target):
    diff = predicted - target
    return torch.mean(diff * torch.tanh(diff) - torch.log(torch.tensor(2.0)) + torch.log(1+torch.exp(-2*torch.abs(diff))))
def loss_fn(predicted_camera_poses, target_camera_poses):
    rmse_loss = rmse_loss_fn(predicted_camera_poses, target_camera_poses)
    logcosh_loss = logcosh_loss_fn(predicted_camera_poses, target_camera_poses)
    geodesic_loss = 0
    low_dist_coeffs = 0
    similar_fc = 0
    centered_pp = 0
    for i in range(predicted_camera_poses.shape[0]):
        m1 = predicted_camera_poses[i, :,:9].reshape(-1, 3, 3)
        m2 = predicted_camera_poses[i, :,:9].reshape(-1, 3, 3)
        geodesic_loss += compute_geodesic_distance_from_two_matrices(m1, m2).mean()
        low_dist_coeffs += torch.sum(torch.abs(predicted_camera_poses[i, :, 16:21]))
        similar_fc += torch.mean((predicted_camera_poses[i, :, 12] - predicted_camera_poses[i, :, 13])**2)
        centered_pp += torch.mean((predicted_camera_poses[i, :, 14] - IMAGE_WIDTH/2)**2 +
                                    (predicted_camera_poses[i, :, 15] - IMAGE_HEIGHT/2)**2)
    projected_points_w_predicted, _ = project_points(fiducials, predicted_camera_poses,
                                                      SMALL_VALUE_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT)
    projected_points_w_target, _ = project_points(fiducials, target_camera_poses,
                                                   SMALL_VALUE_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT)
    reprojection_error = rmse_loss_fn(projected_points_w_predicted, projected_points_w_target)
    return rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error

def save_checkpoint(model, optimizer, epoch, filename):
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'epoch': epoch}
    torch.save(state, filename)
def load_checkpoint(model, optimizer, filename):
    global STARTING_EPOCH
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    STARTING_EPOCH = checkpoint['epoch']
    print(f'Starting epoch: {STARTING_EPOCH}')

# --- Training loop as generator ---
def run_training(calib_obj_file, oem_params_file):
    global stop_training, fiducials
    stop_training = False
    try:
        fiducials_df = pd.read_csv(calib_obj_file.name)
        oem_params_df = pd.read_csv(oem_params_file.name)
    except Exception as e:
        yield (text_to_figure(f"CSV error: {e}"),
               text_to_figure(f"CSV error: {e}"))
        return
    if 'Camera ID' in oem_params_df.columns:
        oem_params_df = oem_params_df.drop(columns=['Camera ID'])
    initial_configs = torch.tensor(oem_params_df.values).float().to(device)
    NUMBER_OF_CAMERAS = len(initial_configs)
    fiducials = torch.tensor(fiducials_df.values).float().to(device)
    global NUMBER_OF_FIDUCIALS
    NUMBER_OF_FIDUCIALS = len(fiducials)
    FIDUCIAL_CENTROID = np.mean(fiducials.cpu().numpy(), axis=0)
    RADIUS = 850.138916015625

    model = DGCCNet().to(device)
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATES[0]},
        {'params': model.R.parameters(), 'lr': LEARNING_RATES[1]},
        {'params': model.t.parameters(), 'lr': LEARNING_RATES[2]},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATES[3]},
        {'params': model.pp.parameters(), 'lr': LEARNING_RATES[4]},
        {'params': model.kc.parameters(), 'lr': LEARNING_RATES[5]}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, factor=0.95)
    writer = SummaryWriter(f'logs/DGCC_{timestamp_str}_{EPOCHS}_{LEARNING_RATES}')
    best_loss = float('inf')
    
    camera_pose_synth = CameraPoseSynthesizer(
        batch_size=BATCH_SIZE,
        fiducials=fiducials,
        OEM_params=initial_configs,
        radius=RADIUS,
        variations=[(torch.rand(1)*INTRINSIC_VARIATION).item(),
                    (torch.rand(1)*EXTRINSIC_VARIATION).item()],
        number_of_cameras=NUMBER_OF_CAMERAS,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        small_value_scale=SMALL_VALUE_SCALE,
        random_angles=True
    )

    train_epochs, train_losses = [], []
    update_interval = 1  # update every epoch

    for epoch in range(STARTING_EPOCH, EPOCHS):
        if stop_training:
            yield (text_to_figure(f"Stopped at epoch {epoch}."),
                   text_to_figure(f"Stopped at epoch {epoch}."))
            break
        optimizer.zero_grad()
        camera_poses, expanded_camera_poses, projected_points = camera_pose_synth.random_poses()
        if torch.isnan(projected_points).any():
            yield (text_to_figure(f"NaN in projected points at epoch {epoch}."),
                   text_to_figure(f"NaN in projected points at epoch {epoch}."))
            break
        proj_mean = projected_points.mean()
        proj_std = projected_points.std()
        normalized = (projected_points - proj_mean) / proj_std
        predicted = model(normalized)
        if torch.isnan(predicted).any():
            yield (text_to_figure(f"NaN in predictions at epoch {epoch}."),
                   text_to_figure(f"NaN in predictions at epoch {epoch}."))
            break
        expanded_predicted = expand_parameters(predicted, SMALL_VALUE_SCALE)
        rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reproj = loss_fn(expanded_predicted, expanded_camera_poses)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        if epoch < WARMUP_EPOCHS:
            total_loss = 100 * rmse_loss + 100 * geodesic_loss
        else:
            total_loss = 100 * rmse_loss + 0.01 * reproj + 100 * geodesic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)
        writer.add_scalar('Loss/total_loss', total_loss.item(), epoch)
        writer.add_scalar('Loss/rmse_loss', rmse_loss.item(), epoch)
        writer.add_scalar('Loss/logcosh_loss', logcosh_loss.item(), epoch)
        writer.add_scalar('Loss/geodesic_loss', geodesic_loss.item(), epoch)
        writer.add_scalar('Loss/low_dist_coeffs', low_dist_coeffs.item(), epoch)
        writer.add_scalar('Loss/similar_fc', similar_fc.item(), epoch)
        writer.add_scalar('Loss/centered_pp', centered_pp.item(), epoch)
        writer.add_scalar('Loss/reprojection_error', reproj.item(), epoch)
        writer.add_scalar('Loss/l1_norm', l1_norm.item(), epoch)
        train_epochs.append(epoch)
        train_losses.append(total_loss.item())
        lr_str = ", ".join([f"{g['lr']:.6f}" for g in optimizer.param_groups])
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss.item()}, LR: {lr_str}")
        if epoch % update_interval == 0:
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(x=train_epochs, y=train_losses,
                                           mode='lines+markers', name='Loss'))
            fig_train.update_layout(title=f"Training loss (Epoch {epoch})",
                                    xaxis_title="Epoch", yaxis_title="Loss")
            fig_cam = go.Figure()
            cam_config = dict(up=dict(x=0, y=0, z=1),
                              center=dict(x=0, y=0, z=-0.2),
                              eye=dict(x=1, y=1, z=1))
            fig_cam.update_layout(margin=dict(l=0,r=0,b=0,t=0),
                                  scene=dict(
                                      xaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
                                      yaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
                                      zaxis=dict(range=[-200, 1.2*RADIUS+20], autorange=False),
                                      aspectmode='manual',
                                      aspectratio=dict(x=1, y=1, z=0.6)
                                  ),
                                  scene_camera=cam_config)
            draw_utilities.draw_points(fig_cam, fiducials.cpu().numpy(), draw_utilities.blue_variations_hex)
            predicted_pose = expanded_predicted[0]
            gt_pose = expanded_camera_poses[0]
            draw_utilities.draw_cameras(fig_cam, predicted_pose, draw_utilities.red_variations_hex, forward_z_direction=-1)
            draw_utilities.draw_cameras(fig_cam, gt_pose, draw_utilities.blue_variations_hex, forward_z_direction=-1)
            draw_utilities.draw_hemisphere(fig_cam, FIDUCIAL_CENTROID, RADIUS)
            yield (fig_train, fig_cam)
            time.sleep(0.1)
    writer.close()
    yield (text_to_figure("Training completed."), text_to_figure("Training completed."))

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## Neural Recalibration Training Demo")
    with gr.Row():
        calib_obj_input = gr.File(label="Calibration Object CSV (Fiducials)")
        oem_params_input = gr.File(label="OEM Calibration Parameters CSV")
    with gr.Row():
        output_train = gr.Plot(label="Training loss")
    with gr.Row():
        output_camera = gr.Plot(label="Prediction vs Ground truth")
    with gr.Row():
        start_button = gr.Button("Start Training")
        stop_button = gr.Button("Stop Training")
    status_text = gr.Textbox(label="Status", interactive=False)
    start_button.click(run_training, inputs=[calib_obj_input, oem_params_input],
                       outputs=[output_train, output_camera])
    stop_button.click(stop_training_fn, inputs=None, outputs=status_text)

demo.launch()
