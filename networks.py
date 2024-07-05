#########################################################################################################################################
# This work is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/  #
# Author: Charalambos Poullis                                                                                                           #
# Contact: https://poullis.org                                                                                                          #
#########################################################################################################################################
# Creation date: 2023/10/25 12:10
#--------------------------------

################################ BOILERPLATE CODE ################################
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

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
import numpy
import matplotlib.pyplot as plt

# A linear layer
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, _bias=True, _ReLU=True, _BatchNorm=False, _LayerNorm=False, _GroupNorm=False, num_groups=None, _Residual=False):
        super(MLP, self).__init__()

        # layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, output_dim, bias=_bias))

        if _BatchNorm:
            self.layers.append(torch.nn.BatchNorm1d(output_dim))
        elif _LayerNorm:
            self.layers.append(torch.nn.LayerNorm(output_dim))
        elif _GroupNorm:
            if num_groups is None:
                num_groups = 1  # Default to 1 if not specified, but it's better to specify an appropriate value.
            self.layers.append(torch.nn.GroupNorm(num_groups, output_dim))

        if _ReLU:
            self.layers.append(torch.nn.ReLU())

        self.residual = None
        if _Residual:
            # dimension matching layer
            self.residual = torch.nn.Linear(input_dim, output_dim)

        # unpack
        self.layers = torch.nn.Sequential(*self.layers)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                # torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.xavier_normal_(layer.weight)

                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        if self.residual is not None:
            # torch.nn.init.kaiming_normal_(self.residual.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.xavier_normal_(self.residual.weight)

            if self.residual.bias is not None:
                self.residual.bias.data.fill_(0.0)

    def forward(self, x_in):
        x_out = self.layers(x_in)

        if self.residual is not None:
            x_res = self.residual(x_in)
            x_out = x_out + x_res

        return x_out

# A stack of linear layers
class MLPStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, _bias=True, _ReLU=True, _BatchNorm=False, _LayerNorm=False, _GroupNorm=False, num_groups=None, _Residual=False):
        super(MLPStack, self).__init__()

        # Initialize ModuleList for layers
        self.layers = torch.nn.ModuleList()

        # First layer
        self.layers.append(MLP(input_dim, hidden_dim, _bias=_bias, _ReLU=_ReLU, _BatchNorm=_BatchNorm, _LayerNorm=_LayerNorm, _GroupNorm=_GroupNorm, num_groups=num_groups, _Residual=_Residual))

        # Intermediate layers
        for i in range(1, max(1, num_layers - 1)):  # Ensure at least one layer is created if num_layers > 1
            self.layers.append(MLP(hidden_dim, hidden_dim, _bias=_bias, _ReLU=_ReLU, _BatchNorm=_BatchNorm, _LayerNorm=_LayerNorm, _GroupNorm=_GroupNorm, num_groups=num_groups, _Residual=_Residual))

        # Final layer - typically without ReLU and normalization to allow for different types of output (e.g., logits for classification)
        self.layers.append(MLP(hidden_dim, output_dim, _bias=_bias, _ReLU=False, _BatchNorm=False, _LayerNorm=False, _GroupNorm=False, _Residual=_Residual))

        # Convert ModuleList to Sequential for easier forward propagation
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x_in):
        # Forward propagation through the sequential layers
        x_out = self.layers(x_in)
        return x_out