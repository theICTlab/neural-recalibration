#########################################################################################################################################
# This work is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/  #
# Author: Charalambos Poullis                                                                                                           #
# Contact: https://poullis.org                                                                                                          #
#########################################################################################################################################
# Creation date: 2023/10/26 15:38
#--------------------------------

################################ BOILERPLATE CODE ################################
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# Load my common files
import sys
sys.path.insert(1, '/home/charalambos/Documents/CODE/Common')

import utilities

# Clear the console
utilities.clear_console()

# clean up previous stuff
torch.cuda.empty_cache()

# initialize the seed
torch.manual_seed(1234)

utilities.set_print_mode('DEBUG')

# check if there is a GPU or CPU
number_of_devices = torch.cuda.device_count()
utilities.cprint(f'Number of GPU devices: {number_of_devices}', type='DEBUG')
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
utilities.cprint(f'Using {device}', type='DEBUG')
###################################################################################
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation
import itertools

blue_variations_hex = ['#ADD8E6', '#4682B4', '#4169E1', '#0000CD', '#00008B']
red_variations_hex = ['#FFA07A', '#FF4500', '#DC143C', '#B22222', '#8B0000']


def to_transformation_matrix(R, t):
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def draw_camera(fig, R, t, color):
    # Draw camera origin
    fig.add_trace(go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]],
                        mode='markers',
                        marker=dict(size=10, color=color), showlegend=False))
    
    # Draw camera axes
    axis_length = 20  # Length of the axis
    for j, color in enumerate(['red', 'green', 'blue']):
        axis = R[:, j] * axis_length + t
        fig.add_trace(go.Scatter3d(x=[t[0], axis[0]], y=[t[1], axis[1]], z=[t[2], axis[2]],
                                mode='lines',
                                line=dict(width=5, color=color), showlegend=False))
        
def draw_camera(fig, R, t, color, forward_z_direction=-1):
    # Draw camera origin
    fig.add_trace(go.Scatter3d(x=[t[0]], y=[t[1]], z=[t[2]],
                        mode='markers',
                        marker=dict(size=10, color=color), showlegend=False))
    
    # Draw camera axes
    axis_length = 200#1#20  # Length of the axis
    arrow_length = 20#0.1#2  # Length of arrowheads
    
    for j, color in enumerate(['red', 'green', 'blue']):
        axis = R[:, j] * axis_length + t
        fig.add_trace(go.Scatter3d(x=[t[0], axis[0]], y=[t[1], axis[1]], z=[t[2], axis[2]],
                                mode='lines',
                                line=dict(width=5, color=color), showlegend=False))

        # Adding arrowheads
        arrow_head = R[:, j] * (axis_length - arrow_length) + t
        for arr in [arrow_head + arrow_length * R[:, (j+1) % 3] / 5, arrow_head - arrow_length * R[:, (j+1) % 3] / 5]:
            fig.add_trace(go.Scatter3d(x=[axis[0], arr[0]], y=[axis[1], arr[1]], z=[axis[2], arr[2]],
                                       mode='lines',
                                       line=dict(width=5, color=color), showlegend=False))
        
    # Draw pyramid (camera frustum)
    pyramid_height = 200#2#20
    pyramid_base = 100#1#10   
    # Define pyramid corners in camera's local coordinate system
    local_corners = np.array([
        [pyramid_base/2, pyramid_base/2, forward_z_direction*pyramid_height],
        [-pyramid_base/2, pyramid_base/2, forward_z_direction*pyramid_height],
        [-pyramid_base/2, -pyramid_base/2, forward_z_direction*pyramid_height],
        [pyramid_base/2, -pyramid_base/2, forward_z_direction*pyramid_height]
    ]).T


    # Transform to global coordinate system
    global_corners = R @ local_corners + t[:, np.newaxis]

    # Draw lines from camera origin to pyramid corners
    for i in range(4):
        corner = global_corners[:, i]
        fig.add_trace(go.Scatter3d(x=[t[0], corner[0]], y=[t[1], corner[1]], z=[t[2], corner[2]],
                                   mode='lines',
                                   line=dict(width=2, color='black'), showlegend=False))

    # Draw lines connecting the pyramid corners to form the base
    for i in range(4):
        for j in range(i+1, 4):
            fig.add_trace(go.Scatter3d(x=[global_corners[0, i], global_corners[0, j]], 
                                       y=[global_corners[1, i], global_corners[1, j]],
                                       z=[global_corners[2, i], global_corners[2, j]],
                                       mode='lines',
                                       line=dict(width=2, color='black'), showlegend=False))



def draw_points(fig, points, colors):

    # Ensure colors cycle if there are more points than colors
    cyclic_colors = itertools.cycle(colors)  
    
    # Take only as many colors as there are points, cycling through colors if necessary
    colors_for_points = [next(cyclic_colors) for _ in range(len(points))]
    
    # Convert to an array of RGBA values (assuming an alpha value of 1.0 for all)
    rgba_colors = [f'rgba({r}, {g}, {b}, 1.0)' for r, g, b in colors_for_points]
    
    # Draw 3D points
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                               mode='markers',
                               marker=dict(size=4, color=rgba_colors, opacity=0.75), showlegend=False))


def draw_cameras(fig, camera_parameters, colors, forward_z_direction=-1):
    rotation_matrices = camera_parameters[:, :9].reshape(-1, 3, 3).detach().cpu().numpy()
    translations = camera_parameters[:, 9:12].detach().cpu().numpy()
    focal_lengths = camera_parameters[:, 12:14]
    principal_points = camera_parameters[:, 14:16]
    distortion_coeffs = camera_parameters[:, 16:]


    for i in range(len(camera_parameters)):
        R_world = Rotation.from_matrix(rotation_matrices[i])
        # R_world = R_world.inv()
        t_world = translations[i] #R_world.apply(translations[i])
        
        if i == 0:
            draw_camera(fig, R_world.as_matrix(), t_world, 'yellow', forward_z_direction)
        else:
            draw_camera(fig, R_world.as_matrix(), t_world, colors[(i-1)%len(colors)], forward_z_direction)
       

def draw_hemisphere(fig, centroid, radius):

    phi = np.linspace(0, np.pi/2, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)

    x = centroid[0] + radius * np.sin(phi) * np.cos(theta)
    y = centroid[1] + radius * np.sin(phi) * np.sin(theta)
    z = centroid[2] + radius * np.cos(phi)

    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale='Viridis', showscale=False, showlegend=False))
 
    # Drawing lines along latitude
    for i in range(x.shape[0]):
        fig.add_trace(go.Scatter3d(x=x[i,:], y=y[i,:], z=z[i,:], mode='lines', line=dict(color='grey', width=1), showlegend=False))
    
    # Drawing lines along longitude
    for i in range(x.shape[1]):
        fig.add_trace(go.Scatter3d(x=x[:,i], y=y[:,i], z=z[:,i], mode='lines', line=dict(color='grey', width=1), showlegend=False))


if __name__ == "__main__":
        RADIUS = 850
        import torch
        camera_poses = torch.load('camera_poses.pt')

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


        # draw the ground truth poses
        draw_cameras(fig, camera_poses, blue_variations_hex)
        # draw the hemisphere I use for the random poses
        draw_hemisphere(fig, [12.257065, 11.355095,  -0.1651535], RADIUS)

        # Show plot
        fig.write_html("cameras.html")
        # fig.write_image("cameras/epoch_{:06d}.png".format(epoch))
        fig.write_image("cameras.png")