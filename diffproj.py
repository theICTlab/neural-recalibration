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
# sys.path.insert(1, '/home/charalambos/Documents/CODE/Common')

# import utilities
import networks
import draw_utilities

# Clear the console
# utilities.clear_console()

# clean up previous stuff
torch.cuda.empty_cache()

# initialize the seed
torch.manual_seed(42)

# utilities.set_print_mode('DEBUG')

# check if there is a GPU or CPU
number_of_devices = torch.cuda.device_count()
print(f'Number of GPU devices: {number_of_devices}')
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
print(f'Using {device}')
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

epsilon = 1e-5

# 2. Camera projection function
def project_points(points, expanded_params, small_value_scale, image_width, image_height, verbose=True):
    # expanded_params [B, Nc, 21]

    batch_projected_points = []
    not_all_visible = False
    for i in range(expanded_params.shape[0]):
        instance_projected_points = []
        for j in range(expanded_params.shape[1]):
            # Scale the small values
            params = torch.cat([expanded_params[i, j, 0:9] / small_value_scale, expanded_params[i, j, 9:16], expanded_params[i, j, 16:22] / small_value_scale])

            # Extract parameters for this instance of multi-camera system
            r1, r2, r3, r4, r5, r6, r7, r8, r9, tx, ty, tz, fc1, fc2, pp1, pp2, kc1, kc2, kc3, kc4, kc5 = params

            R = torch.tensor([
                    [r1, r2, r3], 
                    [r4, r5, r6], 
                    [r7, r8, r9]
                ], requires_grad=True).to(device)

            T = torch.tensor([tx, ty, tz], dtype=torch.float32, requires_grad=True).to(device)
            
            K = torch.tensor([
                [fc1, 0, pp1],
                [0, fc2, pp2],
                [0, 0, 1]
            ], dtype=torch.float32, requires_grad=True).to(device)
            
            # Project points to the camera coordinate system
            # Inverse rotation is the transpose of R
            R_inv = R.t()  
            
            # Subtract translation T from each point in points
            # points is [NUMBER_OF_FIDUCIALS, 3], T is [3], so we need to align their shapes for broadcasting
            points_minus_t = points - T[None, :]

            # Perform matrix multiplication
            # R_inv is [3, 3], P_minus_t.T is [3, 8], resulting in a [3, 8] matrix
            P = torch.mm(R_inv, points_minus_t.T)
            
            # Create a boolean mask to identify points behind the camera
            mask_behind_camera = P[2, :] > 0
            
            # If there are points behind the camera
            if verbose and torch.any(mask_behind_camera):
                print(f"Batch {i}, Camera {j}: At least one point is behind the camera.")
            
            # Project points to the image plane using the intrinsic matrix
            P_image = torch.mm(K, P)

            # Get the final 2D coordinates in the image
            X, Y = -1.0, -1.0
            if (torch.abs(P_image[2, :]) > epsilon).any():
                X = P_image[0, :] / P_image[2, :]
                Y = P_image[1, :] / P_image[2, :]
            else:
                print('Z is less than epsilon.')
                exit()

            # Normalize and center coordinates around the principal point
            X_normalized = (X - pp1) / (fc1)# + epsilon)
            Y_normalized = (Y - pp2) / (fc2)# + epsilon)
        
            # Apply radial and tangential distortion
            r_2 = X_normalized**2 + Y_normalized**2
            radial = 1 + kc1 * r_2 + kc2 * r_2**2 + kc5 * r_2**3
            dx = 2*kc3*X_normalized*Y_normalized + kc4*(r_2 + 2*X_normalized**2)
            dy = kc3*(r_2 + 2*Y_normalized**2) + 2*kc4*X_normalized*Y_normalized
            X_distorted = X_normalized * radial + dx
            Y_distorted = Y_normalized * radial + dy
            
            # Unnormalize and decenter coordinates to get the final coordinates on the image
            X_distorted = X_distorted * fc1 + pp1
            Y_distorted = Y_distorted * fc2 + pp2
 
            distorted_points = torch.vstack([image_width-1-X_distorted, Y_distorted]).T
            
            # Check if x-coordinates are in the range
            x_in_range = (distorted_points[:, 0] > 0) & (distorted_points[:, 0] < image_width)

            # Check if y-coordinates are in the range
            y_in_range = (distorted_points[:, 1] > 0) & (distorted_points[:, 1] < image_height)

            points_in_range = x_in_range & y_in_range
            all_points_in_range = torch.all(points_in_range)

            not_all_visible = not_all_visible or (not all_points_in_range)

            instance_projected_points.append(distorted_points)
        instance_projected_points = torch.stack(instance_projected_points)

        batch_projected_points.append(instance_projected_points)
    batch_projected_points = torch.stack(batch_projected_points)

    return batch_projected_points, not_all_visible # [B, Nc, Nf, 2], True/False
