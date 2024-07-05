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
import timm
sys.stdout = sys.__stdout__
import plotly.graph_objects as go
from utils import *
from diffproj import *
from camera_pose_synthesis import * 

##you can choose the gpu id if you running on a machine with multiple-gpus
# gpu_id = 0
# torch.cuda.set_device(gpu_id)

# Constants
IMAGE_WIDTH = 1600  # Define your image width
IMAGE_HEIGHT = 1600  # Define your image height
EPOCHS = 300000 # Define the number of epochs
WARMUP_EPOCHS = 1000
STARTING_EPOCH = 0
NUMBER_OF_CAMERAS = 10
NUMBER_OF_FIDUCIALS = None
INTRINSIC_VARIATION = 0.1 # % from the original values
EXTRINSIC_VARIATION = 0.1
epsilon = 1e-5 # In projection function, add to denominator to avoid division by 0
LEARNING_RATES = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001] # The learning rates [encoder, R, t, fc, pp, kc]
POWER = 1
SMALL_VALUE_SCALE = 1000
BATCH_SIZE = 2          # select the  batch-size: you may get cuda out of memory with larger batch_sizes
TEST_EPOCHS = 20000     # Number of epochs in testing
# Initialize parser
parser = argparse.ArgumentParser()   
# Adding optional argument
parser.add_argument("-m", "--mode", help = "Set the 'train' or 'test' mode") 
parser.add_argument("-llb", "--load_last_best", help = "Load the last best checkpoint (y|N)") 
# Read arguments from command line
args = parser.parse_args()

# Format the current time as a string in the format: YYYY-MM-DD-HH-MM-SS
timestamp_str = utilities.get_timestamp()

param_names = [
    "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", 
    "tx", "ty", "tz", 
    "fc1", "fc2", 
    "pp1", "pp2", 
    "kc1", "kc2", "kc3", "kc4", "kc5"
]


# 1. Load data: camera configs and 3d fiducials groundtruth
initial_configs = pd.read_csv("camera_parameters_10.csv")
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
RADIUS = 850.138916015625 # you can tune this radius to your specific use
utilities.cprint(f'Hemisphere radius: {RADIUS}', type="DEBUG")


# 2. Network
class DGCCNet(nn.Module):
    def __init__(self):
        super(DGCCNet, self).__init__()
        # Use the pretrained ViT for encoding the images
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Replace the head (classifier) with an Identity layer
        self.encoder.head = nn.Identity()

        # Get the number of output channels from the last block of the ViT model
        self.encoder_head_dim = self.encoder.embed_dim

        # Additional bottleneck layer
        self.bottleneck1 = nn.Sequential(
            nn.Linear(self.encoder_head_dim, 128),
            nn.ReLU(),
        )

        # Use different heads for each parameter
        self.R = self._get_head(128, 6)
        self.t = self._get_head(128, 3)
        self.fc = self._get_head(128, 2)
        self.pp = self._get_head(128, 2)
        self.kc = self._get_head(128, 5)


    def _get_head(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x_in):
        x_in = x_in.view(-1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        x_in = F.interpolate(x_in, size=(224, 224), mode='bilinear', align_corners=False)   #resize to 224*224
        x_enc = self.encoder(x_in)  # Pass the input through the encoder
        x_enc = self.bottleneck1(x_enc)  # Pass through the second bottleneck layer
        x_enc = x_enc.view(BATCH_SIZE, NUMBER_OF_CAMERAS, -1)
        # Pass through each head
        x_R = self.R(x_enc)
        x_t = self.t(x_enc)
        x_fc = self.fc(x_enc)
        x_pp = self.pp(x_enc)
        x_kc = self.kc(x_enc)

        # Concatenate
        x_out = torch.cat([x_R, x_t, x_fc, x_pp, x_kc], dim=2)

        return x_out

def create_color_images(projected_points, gt_images=None):

    # Create a list to store the individual images
    images = []
    # Loop over each set of points (from each camera) in the batch
    visible_from_all = []
    for batch_idx in range(projected_points.shape[0]):
        # Initialize visibility for this batch
        visible_from_all_batch = True

        # Loop over each set of points (from each camera) in the batch
        for i in range(projected_points.shape[1]):
            if gt_images is None:
                # Create a blank image
                img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
            else:
                img = gt_images[batch_idx, i, :, :].numpy().transpose(1, 2, 0)

            # Get the points from the current camera
            points = projected_points[batch_idx, i].cpu().numpy()

            # Loop over each point and mark it on the image
            for j, (x, y) in enumerate(points):
                # Check if the point falls within the bounds of the image
                if 0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT:
                    # Mark the point on the image with a circle of radius 5
                    if gt_images is None: 
                        cv2.circle(img, (int(x), int(y)), 20, bgr_colors[j], -1)
                    else:
                        cv2.circle(img, (int(x), int(y)), 10, bgr_colors[j], -1)
                else:
                    visible_from_all_batch = False

            # Append the image to the list of images
            images.append(img)
        
        # Append visibility for this batch
        visible_from_all.append(visible_from_all_batch)

    # Convert the list of images to a tensor
    image_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
    # Permute the dimensions to get the correct shape
    image_tensor_permuted = image_tensor.permute(0, 3, 1, 2).view(BATCH_SIZE, NUMBER_OF_CAMERAS, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    return image_tensor_permuted, all(visible_from_all)



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

def save_checkpoint(model, optimizer, epoch, filename='best_model_checkpoint_ib.pth.tar'):
    # Save checkpoint including optimizer state
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='best_model_checkpoint_ib.pth.tar'):
    # Load checkpoint and optimizer state
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


#Define the model
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10000, factor=0.95)


if 'y' in args.load_last_best:
    # Load a checkpoint
    load_checkpoint(model, optimizer, 'best_model_checkpoint_ib.pth.tar')
    utilities.cprint('Loaded the last best checkpoint', type="INFO")

# Indices for each group in the tensor:
r_indices = list(range(0, 6))
t_indices = list(range(6, 9))
fc_indices = list(range(9, 11))
pp_indices = list(range(11, 13))
kc_indices = list(range(13, 18))

groups = {
    'r': r_indices,
    't': t_indices,
    'fc': fc_indices,
    'pp': pp_indices,
    'kc': kc_indices
}

#4. Training
if 'train' in args.mode:
    camera_pose_synth = CameraPoseSynthesizer(batch_size=BATCH_SIZE, fiducials=fiducials, OEM_params=initial_configs, radius=RADIUS, 
                                              variations=[INTRINSIC_VARIATION, EXTRINSIC_VARIATION], number_of_cameras=NUMBER_OF_CAMERAS, image_width=IMAGE_WIDTH, 
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
    
        network_input_images, visible_from_all = create_color_images(projected_points.clone().detach().cpu())

        # Step 2: normalize input using image's std and mean
        network_input_images_mean = network_input_images.mean()
        network_input_images_std = network_input_images.std()
        normalized_network_input = (network_input_images - network_input_images_mean) / network_input_images_std
        normalized_network_input = normalized_network_input.to(device)



        # Step 3: Feed the rendered images into the model
        predicted_camera_poses = model(normalized_network_input) # [B, Nc, Nf, 2] -> [B, Nc, 18]
        if torch.isnan(predicted_camera_poses).any():
            utilities.cprint(f"\nNaN encountered in predicted values: \n{predicted_camera_poses}", type='CRITICAL')
            break
        
        # Step 4: Calculate the loss; expand the parameters first to 21
        expanded_predicted_camera_poses = expand_parameters(predicted_camera_poses, SMALL_VALUE_SCALE) #[B, Nc, 21]
        rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error = loss_fn(expanded_predicted_camera_poses, expanded_camera_poses)

        # Step 5: Perform backpropagation and update the network weights
        # Add L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        if epoch < WARMUP_EPOCHS:    #for warmup training phase
        # with torch.autograd.detect_anomaly():
            total_loss = 100*rmse_loss + 100*geodesic_loss
        else:
            total_loss = 100*rmse_loss + 0.01*reprojection_error + 100*geodesic_loss

        total_loss.backward()    
      
        # clip gradients
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        optimizer.step()
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

        if epoch > WARMUP_EPOCHS and best_loss < total_loss :
            best_loss = total_loss            
            save_checkpoint(model, optimizer, epoch, 'best_model_checkpoint_ib.pth.tar')

        # # Get the gradients
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         writer.add_scalar(f"gradients/{name}", param.grad.abs().mean().item(), epoch)

        # Form a single string with all learning rates
        lr_str = ", ".join([f"{param_group['lr']:.6f}" for param_group in optimizer.param_groups])

        utilities.cprint(f'\rEpoch {epoch + 1}/{EPOCHS}, LR: {lr_str}, EXTRINSIC_VARIATION: {EXTRINSIC_VARIATION}, INTRINSIC_VARIATION: {INTRINSIC_VARIATION}, Loss: {total_loss.item()}', type="INFO", end='')

        # # Step 7: Report and visualize every 1000
        # if epoch%50 == 0:
        #     for b in range(BATCH_SIZE):
        #         # Initialize Plotly 3D scatter plot
        #         fig = go.Figure()
        #         camera = dict(
        #             up=dict(x=0, y=0, z=1),
        #             center=dict(x=0, y=0, z=-0.2),
        #             eye=dict(x=1, y=1, z=1)
        #         )
        #         fig.update_layout(
        #             margin=dict(l=0, r=0, b=0, t=0),
        #             # width=700, height=500,
        #             scene=dict(
        #                 xaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
        #                 yaxis=dict(range=[-1.2*RADIUS-20, 1.2*RADIUS+20], autorange=False),
        #                 zaxis=dict(range=[-200, 1.2*RADIUS+20], autorange=False),
        #                 aspectmode='manual',
        #                 aspectratio=dict(x=1, y=1, z=0.6)
        #             ),
        #             scene_camera = camera,
                    
        #         )

        #         # draw the fiducials
        #         draw_utilities.draw_points(fig, fiducials.cpu(), colors)
        #         # draw the predicted cameras
        #         draw_utilities.draw_cameras(fig, expanded_predicted_camera_poses[b], draw_utilities.red_variations_hex, forward_z_direction=-1)
        #         # draw the ground truth poses
        #         draw_utilities.draw_cameras(fig, expanded_camera_poses[b], draw_utilities.blue_variations_hex, forward_z_direction=-1)
        #         # draw the hemisphere I use for the random poses
        #         draw_utilities.draw_hemisphere(fig, FIDUCIAL_CENTROID, RADIUS)

        #         # Show plot
        #         fig.write_html(f"output/multi_camera_system_instance_{b}_.html")
        #         # fig.write_image("cameras/epoch_{:06d}.png".format(epoch))
        #         fig.write_image(f"output/multi_camera_system_instance_{b}_.png")

        #         #Print the prediction/target values; scale down the values
        #         predicted_camera_poses = torch.cat([
        #             expanded_predicted_camera_poses[b, :, 0:9] / SMALL_VALUE_SCALE, 
        #             expanded_predicted_camera_poses[b, :, 9:16], 
        #             expanded_predicted_camera_poses[b, :, 16:22] / SMALL_VALUE_SCALE
        #             ], dim=1)

        #         camera_poses = torch.cat([
        #             expanded_camera_poses[b, :, 0:9] / SMALL_VALUE_SCALE, 
        #             expanded_camera_poses[b, :, 9:16], 
        #             expanded_camera_poses[b, :, 16:22] / SMALL_VALUE_SCALE
        #             ], dim=1)

        #         rotation_matrix = predicted_camera_poses[:,:9].detach().cpu().numpy().reshape(NUMBER_OF_CAMERAS,3,3)
        #         gt_rotation_matrix = camera_poses[:,:9].detach().cpu().numpy().reshape(NUMBER_OF_CAMERAS,3,3)
        #         for cam in range(0,NUMBER_OF_CAMERAS):
        #             utilities.cprint(f"Batch {b}, Camera {cam}:", type="INFO")
        #             rotation_matrix_cam = rotation_matrix[cam,:]

        #             # Check orthogonality
        #             orthogonal_check = np.allclose(np.dot(rotation_matrix_cam, rotation_matrix_cam.T), np.identity(3), atol=1e-6)
        #             utilities.cprint(f"Orthogonality check: {orthogonal_check}", type="WARNING")

        #             # Check determinant
        #             determinant_check = np.allclose(np.linalg.det(rotation_matrix_cam), 1, atol=1e-6)
        #             utilities.cprint(f"Determinant check: {determinant_check}", type="WARNING")

        #             gt_rotation_matrix_cam = gt_rotation_matrix[cam,:]
        #             t_cam  = predicted_camera_poses[cam,9:12].detach().cpu().numpy()
        #             gt_t_cam  = camera_poses[cam,9:12].detach().cpu().numpy()
        #             # r_base_inv = rotation_matrix_cam.dot(np.linalg.inv(r_base))
        #             # if cam == 3 and cv2.Rodrigues(r_base_inv)[1][0][0] > 0.0:
        #             #     print("The Rotation is:", -cv2.Rodrigues(r_base_inv)[0].reshape(3,))
        #             # else:
        #             utilities.cprint(f"Predicted rotation is {Rotation.from_matrix(rotation_matrix_cam).as_rotvec().reshape(3,)}", type="DEBUG")
        #             utilities.cprint(f'{rotation_matrix_cam}', type="DEBUG")
        #             utilities.cprint(f'GT rotation is {Rotation.from_matrix(gt_rotation_matrix_cam).as_rotvec().reshape(3,)}', type="DEBUG")
        #             utilities.cprint(f'{gt_rotation_matrix_cam}', type="DEBUG")

        #             utilities.cprint(f'Predicted translation is {t_cam}', type="DEBUG") #- r_base_inv.dot(t_base))
        #             utilities.cprint(f'GT translation is {gt_t_cam}', type="DEBUG")

        #             if not (orthogonal_check and determinant_check):
        #                 utilities.cprint("Not an orthonormal matrix", type="CRITICAL")
        #                 exit(0)    
    

    writer.close()


if 'test' in args.mode:
    load_checkpoint(model, optimizer, 'best_model_checkpoint_ib.pth.tar')
    
    camera_pose_synth = CameraPoseSynthesizer(batch_size=BATCH_SIZE, fiducials=fiducials, OEM_params=initial_configs, radius=RADIUS, 
                                              variations=[INTRINSIC_VARIATION, EXTRINSIC_VARIATION], number_of_cameras=NUMBER_OF_CAMERAS, image_width=IMAGE_WIDTH, 
                                              image_height=IMAGE_HEIGHT, small_value_scale=SMALL_VALUE_SCALE, random_angles=True)
    
    total_rep_error = 0
    
    
    for epoch in range(TEST_EPOCHS):
        
        # Step 1: Generate random camera poses
        camera_poses, expanded_camera_poses, projected_points = camera_pose_synth.random_poses()
        if torch.isnan(projected_points).any():
            utilities.cprint(f"\nNaN encountered in the projected points: \n{projected_points}", type='CRITICAL')
            break
    

        network_input_images, visible_from_all = create_color_images(projected_points.clone().detach().cpu())

        # Step 2: normalize input using images's std and mean
        network_input_images_mean = network_input_images.mean()
        network_input_images_std = network_input_images.std()
        normalized_network_input = (network_input_images - network_input_images_mean) / network_input_images_std
        normalized_network_input = normalized_network_input.to(device)



        # Step 3: Feed the rendered images into the model
        predicted_camera_poses = model(normalized_network_input) # [B, Nc, Nf, 2] -> [B, Nc, 18]
        if torch.isnan(predicted_camera_poses).any():
            utilities.cprint(f"\nNaN encountered in predicted values: \n{predicted_camera_poses}", type='CRITICAL')
            break
        
        # Step 4: Calculate the loss; expand the parameters first to 21 representation
        expanded_predicted_camera_poses = expand_parameters(predicted_camera_poses, SMALL_VALUE_SCALE) #[B, Nc, 21]
        rmse_loss, logcosh_loss, geodesic_loss, low_dist_coeffs, similar_fc, centered_pp, reprojection_error = loss_fn(expanded_predicted_camera_poses, expanded_camera_poses)
        aa = reprojection_error.item()
        total_rep_error += aa
        print("reprojection error is:", aa)



    print("Mean reprojection error is", total_rep_error/TEST_EPOCHS)