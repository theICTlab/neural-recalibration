#########################################################################################################################################
# This work is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/  #
# Author: Charalambos Poullis                                                                                                           #
# Contact: https://poullis.org                                                                                                          #
#########################################################################################################################################
# Creation date: 2023/09/12 12:02
#--------------------------------

################################ BOILERPLATE CODE ################################
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Load my common files
import sys
sys.path.insert(1, '/home/charalambos/Documents/CODE/Common')

import utilities
import networks
import draw_utilities

# Clear the console
utilities.clear_console()

# clean up previous stuff
torch.cuda.empty_cache()

# initialize the seed
torch.manual_seed(42)

utilities.set_print_mode('DEBUG')

# check if there is a GPU or CPU
number_of_devices = torch.cuda.device_count()
utilities.cprint(f'Number of GPU devices: {number_of_devices}', type='DEBUG')
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
utilities.cprint(f'Using {device}', type='DEBUG')
###################################################################################

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation
import argparse
import math
import sys
sys.stdout = sys.__stdout__
import plotly.graph_objects as go
from utils import *
from diffproj import *
from camera_pose_synthesis import * 

# Constants
IMAGE_WIDTH = 1600  # Define your image width
IMAGE_HEIGHT = 1600  # Define your image height
EPOCHS = 20000 # Define the number of epochs
WARMUP_EPOCHS = 100 
TEST_SIZE = 20000 # Define number of test samples
STARTING_EPOCH = 0
NUMBER_OF_CAMERAS = 10
NUMBER_OF_FIDUCIALS = None
INTRINSIC_VARIATION = 0.05 # % from the original values
EXTRINSIC_VARIATION = 0.05
epsilon = 1e-5 # In projection function, add to denominator to avoid division by 0
LEARNING_RATES = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001] # The learning rates [encoder, R, t, fc, pp, kc]
POWER = 1
SMALL_VALUE_SCALE = 1000
BATCH_SIZE = 512

# Initialize parser
parser = argparse.ArgumentParser()   
# Adding optional argument
parser.add_argument("-m", "--mode", help = "Set the 'train' or 'test' mode") 
parser.add_argument("-llb", "--load_last_best", help = "Load the last best checkpoint (y|N)") 
# Read arguments from command line
args = parser.parse_args()

# Format the current time as a string in the format: YYYY-MM-DD-HH-MM-SS
timestamp_str = utilities.get_timestamp()


# 1. Load data
# initial_configs = pd.read_csv("camera_parameters_10.csv")
initial_configs = pd.read_csv("camera_parameters_curve.csv")
NUMBER_OF_CAMERAS = len(initial_configs)
initial_configs = initial_configs.drop(columns=['Camera ID'])
initial_configs = torch.tensor(initial_configs.values).float().to(device)

fiducials = torch.tensor(pd.read_csv("fiducials_cube_corners.csv").values).float().to(device)
NUMBER_OF_FIDUCIALS = len(fiducials)
utilities.cprint(f'Number of fiducials: {NUMBER_OF_FIDUCIALS}', type="DEBUG")
FIDUCIAL_CENTROID = np.mean(fiducials.cpu().numpy(), axis=0)
utilities.cprint(f'Fiducial centroid: {FIDUCIAL_CENTROID}', type="DEBUG")
FIDUCIAL_EXTENT = np.max(np.linalg.norm(fiducials.cpu().numpy() - FIDUCIAL_CENTROID, axis=1))
CAMERA_CENTROID = np.mean(initial_configs[:,3:6].cpu().numpy(), axis=0)
utilities.cprint(f'Camera centroid: {CAMERA_CENTROID}', type="DEBUG")
RADIUS = 850.138916015625 # This comes from the camcalib_as_opt.py; Basically, you triangulate based on the initial configuration to get an estimate of the radius of the hemisphere 
utilities.cprint(f'Hemisphere radius: {RADIUS}', type="DEBUG")


# 2. Network
class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, nhead=4, num_encoder_layers=4, num_output_transformer_layers=2):
        super(TransformerNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Assuming input_dim is the embedding_dim for the transformer
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Map input to hidden_dim if needed

        # Transformer Encoder as Output Layer
        output_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.output_transformer = nn.TransformerEncoder(output_transformer_layer, num_layers=num_output_transformer_layers)

        # Final Linear Layer to match output_dim
        self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # Embed input to hidden_dim
        x = self.output_transformer(x)  # Further process the Input with the transformer
        output = self.final_linear(x)  # Map to output_dim
        return output
    
class DGCCNet(nn.Module):
    def __init__(self):
        super(DGCCNet, self).__init__()

        # Embed NUMBER_OF_FIDUCIALS*2 points from each camera
        self.embedding = nn.Linear(NUMBER_OF_FIDUCIALS*2, 512)  
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Separate MLPStack heads for parameter prediction
        self.R = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=6, num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.t = networks.MLPStack(input_dim=512,  hidden_dim=256, output_dim=3, num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.fc = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=2, num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.pp = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=2, num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)
        self.kc = networks.MLPStack(input_dim=512, hidden_dim=256, output_dim=5, num_layers=12, _bias=True, _ReLU=True, _LayerNorm=True, _Residual=True)

    def camera_identity_encoding(self, num_cameras, d_model):
        """
        Generates unique encodings for each camera.
        
        Args:
            num_cameras (int): The number of cameras.
            d_model (int): The dimension of the embeddings/encodings.
            
        Returns:
            torch.Tensor: The camera identity encodings (num_cameras, d_model).
        """
        # Simple identity encoding: one-hot encoding + noise to fill d_model dimensions
        identity_encoding = torch.zeros(num_cameras, d_model, device=device)
        for i in range(num_cameras):
            identity_encoding[i, i % d_model] = 1  # Basic one-hot encoding for simplicity
        
        # Optionally, add some small noise or pattern to the rest of the encoding
        # to ensure unique representations if num_cameras > d_model
        noise = torch.randn(num_cameras, d_model, device=device) * 0.01
        identity_encoding += noise
        
        return identity_encoding

    def forward(self, x_in):
        # Assuming x_in shape is [batch_size, NUMBER_OF_CAMERAS, NUMBER_OF_FIDUCIALS*2]
        batch_size, num_cameras, _, _ = x_in.shape
        
        # Keep the batch and camera dimensions separate for embedding
        x_in_flat = x_in.view(-1, NUMBER_OF_FIDUCIALS*2)  # [batch_size * NUMBER_OF_CAMERAS, NUMBER_OF_FIDUCIALS*2]
        x_enc = self.embedding(x_in_flat)  # Process each camera's data
        
        # Generate camera identity encodings
        identity_enc = self.camera_identity_encoding(NUMBER_OF_CAMERAS, 512)
        # Expand identity_enc to match batch size, without flattening the batch
        identity_enc = identity_enc.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, NUMBER_OF_CAMERAS, 512]
        
        # Reshape x_enc to add identity encodings properly
        x_enc = x_enc.view(batch_size, num_cameras, -1)  # Back to [batch_size, NUMBER_OF_CAMERAS, -1]
        x_enc += identity_enc
        
        # Transformer encoder treats each camera within a system as a sequence element
        x = self.encoder(x_enc)  # [batch_size, NUMBER_OF_CAMERAS, 512]
        
        # Process the output for each camera through MLP heads
        # If individual outputs per camera are needed, reshape or aggregate as needed here
        x_flat = x.view(batch_size * num_cameras, -1)  # Flatten for MLP processing

        x_R = self.R(x_flat)
        x_t = self.t(x_flat)
        x_fc = self.fc(x_flat)
        x_pp = self.pp(x_flat)
        x_kc = self.kc(x_flat)

        # Assuming you want individual outputs per camera, concatenate and reshape
        x_out = torch.cat([x_R, x_t, x_fc, x_pp, x_kc], dim=-1)
        x_out = x_out.view(batch_size, num_cameras, -1)  # [batch_size, NUMBER_OF_CAMERAS, output_dims]

        return x_out

# 3. Loss functions
def rmse_loss_fn(predicted, target):
    # predicted [B, Nc, 21]
    # target [B, Nc, 21]

    """Calculate the RMSE loss between predicted and target tensors"""
    mse = torch.nn.functional.mse_loss(predicted, target, reduction='mean')
    rmse = torch.sqrt(mse)
 
    return rmse

def logcosh_loss_fn(predicted, target):
    # predicted [B, Nc, 21]
    # target [B, Nc, 21]
    diff = predicted - target
    return torch.mean(diff * torch.tanh(diff) - torch.log(torch.tensor(2.0)) + torch.log(1 + torch.exp(-2 * torch.abs(diff))))

# Save checkpoint including optimizer state
def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filename)

# Load checkpoint and optimizer state
def load_checkpoint(model, optimizer, filename):
    global STARTING_EPOCH

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    STARTING_EPOCH = checkpoint['epoch']
    utilities.cprint(f'Starting epoch: {STARTING_EPOCH}')

def loss_fn(predicted_camera_poses, target_camera_poses):
    # predicted [B, Nc, 21]
    # target [B, Nc, 21]

    # RMSE on all parameters
    rmse_loss = rmse_loss_fn(predicted_camera_poses, target_camera_poses)

    # Log-cosh for monitoring
    logcosh_loss = logcosh_loss_fn(predicted_camera_poses, target_camera_poses)

    # Geodesic loss for rotation matrices
    geodesic_loss = 0
    low_dist_coeffs = 0
    similar_fc = 0
    centered_pp = 0
    for i in range(predicted_camera_poses.shape[0]):
        m1 = predicted_camera_poses[i, :,:9].reshape(-1, 3, 3)
        m2 = predicted_camera_poses[i, :,:9].reshape(-1, 3, 3)
        geodesic_loss += compute_geodesic_distance_from_two_matrices(m1, m2).mean()

        low_dist_coeffs += torch.sum(torch.abs(predicted_camera_poses[i, :, 16:21]))
        similar_fc += torch.mean((predicted_camera_poses[i, :, 12] - predicted_camera_poses[i, :, 13])**2) # similar fx, fy
        centered_pp += torch.mean((predicted_camera_poses[i, :, 14] - IMAGE_WIDTH/2)**2 + (predicted_camera_poses[i, :, 15] - IMAGE_HEIGHT/2)**2) # centered cx, cy

    # Calculate reprojection_error; rescale the small values
    projected_points_w_predicted, _ = project_points(fiducials, predicted_camera_poses, SMALL_VALUE_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT)
    projected_points_w_target, _ = project_points(fiducials, target_camera_poses, SMALL_VALUE_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT)
    reprojection_error = rmse_loss_fn(projected_points_w_predicted, projected_points_w_target)
    
    return rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error



# Define the model and optimizer
model = DGCCNet()
model = model.to(device)
utilities.cprint(f'Optimized parameters: {utilities.count_model_parameters(model)}', type="INFO")
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': LEARNING_RATES[0]},
    {'params': model.R.parameters(), 'lr': LEARNING_RATES[1]},
    {'params': model.t.parameters(), 'lr': LEARNING_RATES[2]},
    {'params': model.fc.parameters(), 'lr': LEARNING_RATES[3]},
    {'params': model.pp.parameters(), 'lr': LEARNING_RATES[4]},
    {'params': model.kc.parameters(), 'lr': LEARNING_RATES[5]}
])

# Define the ReduceLROnPlateau scheduler for efficient learning
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, factor=0.95)

# Enter the training loop in case train argument is included in the command line
if 'train' in args.mode:
    
    camera_pose_synth = CameraPoseSynthesizer(batch_size=BATCH_SIZE, fiducials=fiducials, OEM_params=initial_configs, radius=RADIUS, 
                                              variations=[(torch.rand(1) * INTRINSIC_VARIATION).item(), (torch.rand(1) * EXTRINSIC_VARIATION).item()], number_of_cameras=NUMBER_OF_CAMERAS, image_width=IMAGE_WIDTH, 
                                              image_height=IMAGE_HEIGHT, small_value_scale=SMALL_VALUE_SCALE, random_angles=True)
    
    # Initialize a variable to hold the best loss
    best_loss = float('inf')

    writer = SummaryWriter(f'logs/DGCC_{timestamp_str}_{EPOCHS}_{LEARNING_RATES}')

    for epoch in range(STARTING_EPOCH, EPOCHS):
        optimizer.zero_grad()
        
        # Step 1: Generate random camera poses
        camera_poses, expanded_camera_poses, projected_points = camera_pose_synth.random_poses()
        if torch.isnan(projected_points).any():
            utilities.cprint(f"\nNaN encountered in the projected points: \n{projected_points}", type='CRITICAL')
            break
    
        # Step 2: normalize input using data's std and mean
        projected_points_mean = projected_points.mean()
        projected_points_std = projected_points.std()
        normalized_projected_points = (projected_points - projected_points_mean) / projected_points_std

        # Step 3: Feed the rendered images into the model
        predicted_camera_poses = model(normalized_projected_points) # [B, Nc, Nf, 2] -> [B, Nc, 18]
        if torch.isnan(predicted_camera_poses).any():
            utilities.cprint(f"\nNaN encountered in predicted values: \n{predicted_camera_poses}", type='CRITICAL')
            break
        
        # Step 4: Calculate the loss; expand the parameters from 18 to 21
        expanded_predicted_camera_poses = expand_parameters(predicted_camera_poses, SMALL_VALUE_SCALE) #[B, Nc, 21]
        rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error = loss_fn(expanded_predicted_camera_poses, expanded_camera_poses)

        # Step 5: Perform backpropagation and update the network weights
        # Add L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        if epoch < WARMUP_EPOCHS:
            total_loss = 100*rmse_loss + 100*geodesic_loss  # 100*logcosh_loss + low_dist_coeffs + similar_fc + centered_pp + large_fc #+ l1_norm + low_dist_coeffs + similar_fc + centered_pp + large_fc
        else:
            total_loss = 100*rmse_loss + 0.01*reprojection_error + 100*geodesic_loss

        total_loss.backward()    
      
        # clip the gradients
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        # Apply learning rate decay every epoch
        scheduler.step(total_loss)

        # Step 6: Log the losses and parameters to TensorBoard
        writer.add_scalar('Loss/total_loss', total_loss, epoch)
        writer.add_scalar('Loss/rmse_loss', rmse_loss, epoch)
        writer.add_scalar('Loss/logcosh_loss', logcosh_loss, epoch)
        writer.add_scalar('Loss/geodesic_loss', geodesic_loss, epoch)
        writer.add_scalar('Loss/low_dist_coeffs', low_dist_coeffs, epoch)
        writer.add_scalar('Loss/similar_fc', similar_fc, epoch)
        writer.add_scalar('Loss/centered_pp', centered_pp, epoch)
        writer.add_scalar('Loss/reprojection_error', reprojection_error, epoch)
        writer.add_scalar('Loss/l1_norm', l1_norm, epoch)

        if torch.isnan(total_loss).any():
            utilities.cprint("\nNaN loss encountered", type='CRITICAL')
            break

        if epoch > WARMUP_EPOCHS and total_loss < best_loss:
            best_loss = total_loss            
            save_checkpoint(model, optimizer, epoch, 'best_model_checkpoint.pth.tar')

        # Keep track of the gradients in summary writer
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         writer.add_scalar(f"gradients/{name}", param.grad.abs().mean().item(), epoch)

        # Form a single string with all learning rates
        lr_str = ", ".join([f"{param_group['lr']:.6f}" for param_group in optimizer.param_groups])

        utilities.cprint(f'\rEpoch {epoch + 1}/{EPOCHS}, LR: {lr_str}, EXTRINSIC_VARIATION: {EXTRINSIC_VARIATION}, INTRINSIC_VARIATION: {INTRINSIC_VARIATION}, Loss: {total_loss.item()}', type="INFO", end='')

        # Step 6: Report and visualize every 1000
        if epoch%1000 == 0:
            # You can avoid plotting all the batch to speed up the process!
            for b in range(BATCH_SIZE):
                # Initialize Plotly 3D scatter plot
                fig = go.Figure()
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=-0.2),
                    eye=dict(x=1, y=1, z=1)
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    # width=700, height=500,
                    scene=dict(
                        xaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
                        yaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
                        zaxis=dict(range=[-200, 1.2*RADIUS+20], autorange=False),
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=0.6)
                    ),
                    scene_camera = camera,
                    
                )

                # draw the fiducials
                draw_utilities.draw_points(fig, fiducials.cpu(), colors)
                # draw the predicted cameras
                draw_utilities.draw_cameras(fig, expanded_predicted_camera_poses[b], draw_utilities.red_variations_hex, forward_z_direction=-1)
                # draw the ground truth poses
                draw_utilities.draw_cameras(fig, expanded_camera_poses[b], draw_utilities.blue_variations_hex, forward_z_direction=-1)
                # draw the hemisphere I use for the random poses
                draw_utilities.draw_hemisphere(fig, FIDUCIAL_CENTROID, RADIUS)

                # Show plot
                fig.write_html(f"output/multi_camera_system_epoch_{epoch}_instance_{b}.html")
                # fig.write_image("cameras/epoch_{:06d}.png".format(epoch))
                fig.write_image(f"output/multi_camera_system_epoch_{epoch}_instance_{b}.png")

                # Print the prediction/target values; scale down the values
                predicted_camera_poses = torch.cat([
                    expanded_predicted_camera_poses[b, :, 0:9] / SMALL_VALUE_SCALE, 
                    expanded_predicted_camera_poses[b, :, 9:16], 
                    expanded_predicted_camera_poses[b, :, 16:22] / SMALL_VALUE_SCALE
                    ], dim=1)

                camera_poses = torch.cat([
                    expanded_camera_poses[b, :, 0:9] / SMALL_VALUE_SCALE, 
                    expanded_camera_poses[b, :, 9:16], 
                    expanded_camera_poses[b, :, 16:22] / SMALL_VALUE_SCALE
                    ], dim=1)

                rotation_matrix = predicted_camera_poses[:,:9].detach().cpu().numpy().reshape(NUMBER_OF_CAMERAS,3,3)
                gt_rotation_matrix = camera_poses[:,:9].detach().cpu().numpy().reshape(NUMBER_OF_CAMERAS,3,3)
                for cam in range(0,NUMBER_OF_CAMERAS):
                    utilities.cprint(f"Batch {b}, Camera {cam}:", type="INFO")
                    rotation_matrix_cam = rotation_matrix[cam,:]

                    # Check orthogonality
                    orthogonal_check = np.allclose(np.dot(rotation_matrix_cam, rotation_matrix_cam.T), np.identity(3), atol=1e-6)
                    utilities.cprint(f"Orthogonality check: {orthogonal_check}", type="WARNING")

                    # Check determinant
                    determinant_check = np.allclose(np.linalg.det(rotation_matrix_cam), 1, atol=1e-6)
                    utilities.cprint(f"Determinant check: {determinant_check}", type="WARNING")

                    gt_rotation_matrix_cam = gt_rotation_matrix[cam,:]
                    t_cam  = predicted_camera_poses[cam,9:12].detach().cpu().numpy()
                    gt_t_cam  = camera_poses[cam,9:12].detach().cpu().numpy()
                    utilities.cprint(f"Predicted rotation is {Rotation.from_matrix(rotation_matrix_cam).as_rotvec().reshape(3,)}", type="DEBUG")
                    utilities.cprint(f'{rotation_matrix_cam}', type="DEBUG")
                    utilities.cprint(f'GT rotation is {Rotation.from_matrix(gt_rotation_matrix_cam).as_rotvec().reshape(3,)}', type="DEBUG")
                    utilities.cprint(f'{gt_rotation_matrix_cam}', type="DEBUG")

                    utilities.cprint(f'Predicted translation is {t_cam}', type="DEBUG") #- r_base_inv.dot(t_base))
                    utilities.cprint(f'GT translation is {gt_t_cam}', type="DEBUG")

                    if not (orthogonal_check and determinant_check):
                        utilities.cprint("Not an orthonormal matrix", type="CRITICAL")
                        exit(0)    
    

    writer.close()


# Enter the test loop in case the test argument is included in the command line
elif 'test' in args.mode:
    
    # Load the best saved checkpoint
    model_name = "best_model_checkpoint.pth.tar"
    load_checkpoint(model, optimizer, model_name)
    utilities.cprint('Loaded the last best checkpoint', type="INFO")
    
    # Keep all samples' reprojection error in a list
    all_reprojection_errors = []

    # Put the model on test mode
    model.eval()

    for epoch in range(TEST_SIZE):
        
        # We use batch_size of 1 for testing
        BATCH_SIZE = 1
        camera_pose_synth = CameraPoseSynthesizer(batch_size=BATCH_SIZE, fiducials=fiducials, OEM_params=initial_configs, radius=RADIUS, 
                                              variations=[(torch.rand(1) * INTRINSIC_VARIATION).item(), (torch.rand(1) * EXTRINSIC_VARIATION).item()], number_of_cameras=NUMBER_OF_CAMERAS, image_width=IMAGE_WIDTH, 
                                              image_height=IMAGE_HEIGHT, small_value_scale=SMALL_VALUE_SCALE, random_angles=True)

        # Step 1: Generate random camera poses
        camera_poses, expanded_camera_poses, projected_points = camera_pose_synth.random_poses()
        if torch.isnan(projected_points).any():
            utilities.cprint(f"\nNaN encountered in the projected points: \n{projected_points}", type='CRITICAL')
            break
    
        # Step 2: normalize input using data's std and mean
        projected_points_mean = projected_points.mean()
        projected_points_std = projected_points.std()
        normalized_projected_points = (projected_points - projected_points_mean) / projected_points_std

        # Step 3: Feed the rendered images into the model
        predicted_camera_poses = model(normalized_projected_points) # [B, Nc, Nf, 2] -> [B, Nc, 18]
        if torch.isnan(predicted_camera_poses).any():
            utilities.cprint(f"\nNaN encountered in predicted values: \n{predicted_camera_poses}", type='CRITICAL')
            break
        
        # Step 4: Calculate the loss; expand the parameters from 18 to 21
        expanded_predicted_poses = expand_parameters(predicted_camera_poses, SMALL_VALUE_SCALE) #[B, Nc, 21]
        rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error = loss_fn(expanded_predicted_poses, expanded_camera_poses)
        all_reprojection_errors.append(reprojection_error.item())
        utilities.cprint(f'\rEpoch {epoch + 1}/{TEST_SIZE}, Reprojection Error is: {reprojection_error.item()}, INTRINSIC_VARIATION: {INTRINSIC_VARIATION}, EXTRINSIC_VARIATION: {EXTRINSIC_VARIATION}', type="INFO", end='')
    

    # Calulating the mean reprojection error over all the test samples
    total_reprojection_error = sum(all_reprojection_errors) / len(all_reprojection_errors)
    utilities.cprint(f'\rMean of error is: {total_reprojection_error}', type="INFO", end='')
 