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
from utils import *
from diffproj import *

epsilon = 1e-5

class CameraPoseSynthesizer:
    def __init__(self, batch_size, fiducials, OEM_params, radius, variations, number_of_cameras, image_width, image_height, small_value_scale=1, random_angles=True) -> None:
        self.batch_size = batch_size
        self.fiducials = fiducials
        self.OEM_params = OEM_params
        self.radius = radius
        self.variations = variations
        self.random_angles = random_angles
        self.number_of_cameras = number_of_cameras
        self.small_value_scale = small_value_scale
        self.image_width = image_width
        self.image_height = image_height
        return

    # Calculates a random point on a hemisphere of the given radius    
    def random_point_on_hemisphere(self):
        # Generate batch_size number of theta and phi values
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi / 2)

        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)
        return np.array([x, y, z])

    def calculate_centroid(self, translations):
        """
        Calculate the centroid of the camera system.
        :param translations: List of translation vectors for each camera (excluding the reference camera).
        :return: Centroid of the camera system.
        """
        if len(translations) == 0:
            return np.array([0, 0, 0])  # If there are no other cameras, the centroid is at the origin

        # Summing up all translations and dividing by number of cameras to get the centroid
        centroid = np.sum(translations, axis=0) / len(translations)
        return centroid

    def orient_camera_system(self, centroid, forward_direction=np.array([0, 0, -1])):
        """
        Calculate the rotation matrix to orient the camera system towards the origin.
        :param centroid: The new centroid of the camera system.
        :param forward_direction: The forward direction vector of the camera system.
        :return: Rotation matrix to orient the camera system towards the origin.
        """
        # Vector pointing from the centroid to the origin
        direction_to_origin = -centroid / np.linalg.norm(centroid)

        # Calculating the rotation needed to align the forward direction with the direction to the origin
        rotation_vector = np.cross(forward_direction, direction_to_origin)
        if np.linalg.norm(rotation_vector) < 1e-6:
            if np.dot(forward_direction, direction_to_origin) > 0:
                # Vectors are nearly parallel, no rotation needed
                return Rotation.from_quat([0, 0, 0, 1])
            else:
                # Vectors are anti-parallel, rotate 180 degrees around an arbitrary perpendicular axis
                perpendicular_vector = np.cross(forward_direction, np.array([1, 0, 0]))
                if np.linalg.norm(perpendicular_vector) < 1e-6:  # if forward_direction is along the x-axis
                    perpendicular_vector = np.cross(forward_direction, np.array([0, 1, 0]))
                perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
                return Rotation.from_rotvec(np.pi * perpendicular_vector)

        rotation_vector_normalized = rotation_vector / np.linalg.norm(rotation_vector)
        rotation_angle = np.arccos(np.clip(np.dot(forward_direction, direction_to_origin), -1.0, 1.0))
        rotation = Rotation.from_rotvec(rotation_angle * rotation_vector_normalized)

        return rotation

    def position_and_orient_camera_system(self, camera_translations, camera_rotations):
        """
        Position and orient the camera system at a random point on a hemisphere with centroid at the random point.
        :param radius: Radius of the hemisphere.
        :param camera_translations: List of translation vectors for each camera (excluding the reference camera).
        :param camera_rotations: List of SciPy Rotation objects for each camera (excluding the reference camera).
        :return: New translations and rotations for each camera in the system.
        """
        # Calculate the current centroid and generate a random point on the hemisphere
        # current_centroid = calculate_centroid(camera_translations)
        random_point = self.random_point_on_hemisphere()

        # Orient the camera system towards the origin
        R_orient = self.orient_camera_system(random_point)
        new_translations = [R_orient.apply(t) for t in camera_translations]
        new_rotations = [R_orient * r for r in camera_rotations]

        # Translation to move the centroid to the random point
        translation_to_random_point = random_point - self.calculate_centroid(new_translations)
        new_translations = [t + translation_to_random_point for t in new_translations]

        # Create and apply a random rotation around the normal vector at the random point
        normal_vector = random_point / np.linalg.norm(random_point)
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_rotation = Rotation.from_rotvec(random_angle * normal_vector)
        new_translations = [random_rotation.apply(t) for t in new_translations]
        new_rotations = [(random_rotation * r) for r in new_rotations]

        return new_translations, new_rotations


    def random_poses(self):
        batch_camera_poses = []
        batch_expanded_camera_poses = []
        batch_projected_points = []
        failure_counter = 0

        # This function will generate self.batch_size instances of mutli-camera systems for which all fiducials will be visible.
        # It will also apply self.variations to the intrinsic and extrinsic parameters
        for _ in range(self.batch_size):

            # Keep repeating this process until all cameras 'see' all the fiducials
            visible_from_all = False
            while not visible_from_all:

                # Apply random extrinsic variations
                camera_translations_variation_tensor = np.random.rand(self.number_of_cameras, 3) * self.variations[0] - (self.variations[0] / 2)
                camera_translations = self.OEM_params[:, 3:6].cpu().numpy() * (camera_translations_variation_tensor + 1)

                camera_rotations_variation_tensor = np.random.rand(self.number_of_cameras, 3) * self.variations[0] - (self.variations[0] / 2)
                camera_rotations = self.OEM_params[:, 0:3].cpu().numpy() * (camera_rotations_variation_tensor + 1)
                camera_rotations = [Rotation.from_euler('xyz', euler_angles, degrees=False) for euler_angles in camera_rotations]

                new_translations, new_rotations = self.position_and_orient_camera_system(camera_translations, camera_rotations)

                instance_multi_cam_poses = []
                for i in range(self.number_of_cameras):
                    (x, y, z) = new_translations[i]
                    rotation_matrix = new_rotations[i]

                    # Apply random intrinsic variations
                    intrinsics = self.OEM_params[i, 6:10].cpu().numpy()
                    intrinsics_variation_tensor = np.random.rand(intrinsics.shape[0]) * self.variations[1] - (self.variations[1] / 2)
                    intrinsics = intrinsics * (intrinsics_variation_tensor + 1)
                    fc1, fc2, pp1, pp2 = intrinsics

                    distortion = self.OEM_params[i, 10:].cpu().numpy()
                    kc_variation_tensor = np.random.rand(distortion.shape[0]) * self.variations[1] - (self.variations[1] / 2)
                    distortion = distortion * (kc_variation_tensor + 1)
                    kc1, kc2, kc3, kc4, kc5 = distortion

                    rotation_raw = compute_ortho6d_from_rotation_matrix(torch.from_numpy(rotation_matrix.as_matrix()).float().unsqueeze(dim=0).to(device)).squeeze(dim=0)
                    # Convert rotation_raw to numpy array
                    rotation_raw_np = np.array(rotation_raw.cpu())

                    # Concatenate all variables
                    # pose_np = np.concatenate([rotation_matrix.as_matrix().ravel(), [x, y, z, fc1, fc2, pp1, pp2, kc1, kc2, kc3, kc4, kc5]])
                    pose_np = np.concatenate([rotation_raw_np, np.array([x, y, z, fc1, fc2, pp1, pp2, kc1, kc2, kc3, kc4, kc5])])
                    pose = torch.tensor(pose_np).float().to(device)
                    
                    # instance_poses.append(torch.tensor(pose_np).float().to(device))
                    instance_multi_cam_poses.append(pose)

                instance_multi_cam_poses = torch.stack(instance_multi_cam_poses) #[NUMBER_OF_CAMERAS, 18]
                
                # Verify that all fiducials are visible 
                expanded_camera_poses = expand_parameters(instance_multi_cam_poses.unsqueeze(dim=0), self.small_value_scale).squeeze(dim=0)
                
                projected_points, not_all_visible = project_points(self.fiducials, expanded_camera_poses.unsqueeze(dim=0), self.small_value_scale, self.image_width, self.image_height, verbose=False)
                projected_points = projected_points.squeeze(dim=0).to(device)
                if torch.isnan(projected_points).any():
                    print(f"\nNaN encountered in the input: \n{projected_points}")
                    break

                # Save the images of the projected points
                # rendered_images, visible_from_all = render_color_images(projected_points.clone().detach().cpu(), self.image_width, self.image_height)
                visible_from_all = not not_all_visible
                if visible_from_all:
                    failure_counter = 0
                else:
                    failure_counter += 1
                    if failure_counter % 100 == 0:
                        print(f'Synthesizing camera poses: {failure_counter} failed attempts.')

            batch_camera_poses.append(instance_multi_cam_poses)  #current instance
            batch_expanded_camera_poses.append(expanded_camera_poses)
            batch_projected_points.append(projected_points)

        return torch.stack(batch_camera_poses), torch.stack(batch_expanded_camera_poses), torch.stack(batch_projected_points) #, rendered_images  # Stack all instances to form the batch



if __name__ == '__main__':
    # 1. Load data
    initial_configs = pd.read_csv("camera_parameters.csv")
    initial_configs = initial_configs.drop(columns=['Camera ID'])
    initial_configs = torch.tensor(initial_configs.values).float().to(device)
    fiducials = torch.tensor(pd.read_csv("fiducials_cube_corners.csv").values).float().to(device)

    # 2. Synthesize camera poses
    camera_pose_synth = CameraPoseSynthesizer(batch_size, fiducials=fiducials, OEM_params=initial_configs, radius = 800, variations=[0.05, 0.05], number_of_cameras=10, image_width=1600, image_height=1600, small_value_scale=1000, random_angles=True)
    camera_poses, expanded_camera_poses, projected_points = camera_pose_synth.random_poses()

    for i in range(0, camera_poses.shape[0]):
        print(f'Batch {i+1}/{camera_poses.shape[0]}')
        for j in range(0, camera_poses.shape[1]):
            print(f'Camera {j+1}/{camera_poses.shape[1]}: {camera_poses[i][j]}')
    
