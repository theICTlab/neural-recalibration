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
import cv2
import numpy as np

epsilon = 1e-5

# Define a list of distinct colors to use for the different points
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 128, 128), (0, 128, 128), (255, 255, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 255, 255), (255, 128, 255), (255, 255, 128),
    (64, 64, 64), (128, 128, 0), (0, 128, 64),
    (128, 0, 128), (0, 128, 128), (128, 0, 0),
    (0, 64, 128), (64, 0, 128), (128, 64, 0),
     (255, 0, 128), (128, 255, 64), (0, 128, 192),
    (192, 128, 255), (255, 192, 128), (128, 128, 0),
    (0, 0, 128), (128, 0, 64), (64, 128, 0),
    (64, 0, 128), (128, 64, 128), (255, 0, 64),
    (64, 255, 128), (0, 64, 255), (192, 0, 255),
    (255, 64, 0), (0, 255, 64), (64, 192, 255),
    (255, 64, 192), (192, 255, 64), (64, 192, 0),
    (255, 128, 64), (64, 255, 192), (192, 64, 255),
    (255, 192, 64), (64, 255, 0), (0, 64, 255),
    (192, 0, 64), (255, 64, 128), (128, 255, 192),
    (192, 255, 128), (255, 128, 192), (128, 192, 255),
    (255, 192, 255), (192, 255, 192), (192, 192, 255),
    (255, 255, 192), (192, 255, 255), (255, 192, 255),
    (192, 255, 192), (192, 192, 255), (255, 255, 192),
    (192, 255, 255), (255, 192, 255), (192, 255, 192),
    (192, 192, 255), (255, 255, 192), (192, 255, 255),
    (255, 192, 255), (192, 255, 192), (192, 192, 255),
    (255, 255, 192), (192, 255, 255), (255, 192, 255),
    (192, 255, 192), (192, 192, 255), (255, 255, 192),
    (192, 255, 255), (255, 192, 255), (192, 255, 192)
]

# Convert RGB to BGR
bgr_colors = [(b, g, r) for (r, g, b) in colors]



def compute_ortho6d_from_rotation_matrix(matrix):
    # Extract the first two columns from the 3x3 rotation matrix
    x = matrix[:, :, 0]  # batch*3
    y = matrix[:, :, 1]  # batch*3
    
    # Normalize the vectors (optional, if they are not already unit vectors)
    x = normalize_vector(x)  # batch*3
    y = normalize_vector(y)  # batch*3
    
    # Concatenate to form the 6D representation
    ortho6d = torch.cat((x, y), 1)  # batch*6
    
    return ortho6d

# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([epsilon]).to(device)))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
      

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    eps = 0.001
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.clamp(cos, min=-1.0+eps, max=1.0-eps)  # Clamp to [-1, 1]
    # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(device)) )
    # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(device))*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def expand_parameters(params, small_value_scale=1):
    #params [B, Nc, 18]
    expanded_params = []
    for i in range(params.shape[0]):
        # Extract the first 6 values from each row
        rotation_raw = params[i, :, :6]
        # Compute the rotation matrix from ortho6d [6,6] -> [6, 3, 3]
        rotation_matrix = compute_rotation_matrix_from_ortho6d(rotation_raw)
        
        # Flatten it: [6,9]
        rotation_matrix_flattened = rotation_matrix.view(-1, 9) 
        
        # Concatenate with the rest: [6, 21]
        reshaped_params = torch.cat([rotation_matrix_flattened, params[i, :, 6:]], dim=1)
        
        # Scale the small values
        rescaled_params = torch.cat([reshaped_params[:, 0:9] * small_value_scale, reshaped_params[:, 9:16], reshaped_params[:, 16:22] * small_value_scale], dim=1)
        
        expanded_params.append(rescaled_params)
    expanded_params = torch.stack(expanded_params)

    return expanded_params #[B, Nc, 21]
        

# Given the input tensor convert to images
def render_color_images(projected_points, image_width, image_height, gt_images=None):
    
    # Create a list to store the individual images
    images = []

    # Loop over each set of points (from each camera)
    visible_from_all = True
    for i in range(projected_points.shape[0]):
        if gt_images is None:
            # Create a blank image
            img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        else:
            img = gt_images[i,:, :, :].numpy().transpose(1, 2, 0)

        # Get the points from the current camera
        points = projected_points[i].cpu().numpy()
        
        # Loop over each point and mark it on the image
        for j, (x, y) in enumerate(points):
            # Check if the point falls within the bounds of the image
            if 0 <= x < image_width and 0 <= y < image_height:
                
                # Mark the point on the image with a circle of radius 5
                if gt_images is None:
                    cv2.circle(img, (int(x), int(y)), 20, colors[j%len(colors)], -1)
                else:
                    cv2.circle(img, (int(x), int(y)), 10, colors[8], -1)
            else:
                visible_from_all = False
        
        # Append the image to the list of images
        images.append(img)
    
    # Convert the list of images to a tensor
    image_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
    # Permute the dimensions to get the correct shape
    image_tensor_permuted = image_tensor.permute(0, 3, 1, 2)
    
    return image_tensor_permuted, visible_from_all

