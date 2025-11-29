"""
Neural network definitions for PDDL
"""

import torch
import torch.nn as nn
from config import *

def weights_init(m):
    """Xavier uniform initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class UnifiedDNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh(), 
                 output_activation=None):
        super().__init__()
        layers = [nn.Linear(dim_in, n_node), activation]
        for _ in range(n_layer):
            layers.extend([nn.Linear(n_node, n_node), activation])
        layers.append(nn.Linear(n_node, dim_out))
        self.net = nn.Sequential(*layers)
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.output_activation = output_activation
        self.net.apply(weights_init)

    def forward(self, x):
        out = self.net(x)
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.output_activation == 'tanh':
            return torch.tanh(out)
        return out

class DNN1(UnifiedDNN):
    """Network for solid temperature prediction"""
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__(dim_in, dim_out, n_layer, n_node, ub, lb, activation)

class DNN2(UnifiedDNN):
    """Network for interface prediction with sigmoid output"""
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__(dim_in, dim_out, n_layer, n_node, ub, lb, activation, output_activation='sigmoid')

class DNN3(UnifiedDNN):
    """Network for fluid temperature prediction"""
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__(dim_in, dim_out, n_layer, n_node, ub, lb, activation)