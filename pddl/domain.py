"""
Domain setup and point generation for PDDL simulation
"""

import numpy as np
import torch
from pyDOE import lhs
from config import *

def delete(MAT, N):
    """Filter points based on geometric constraints"""
    CM = []
    for x in MAT:
        if x[-1] * (P[1] - P[0]) + P[0] > x[0]:
            CM.append(x)
            if len(CM) >= N:
                break
    return np.array(CM)

def generate_domain_points():
    """Generate all training points for the domain"""
    
    # Generate boundary points
    left_wall_s = np.array([x_min, delta, t_min + eps, p_min]) + np.array([0.0, y_max - delta, t_max, p_max]) * lhs(4, int(N_b / 2))
    left_wall_f = np.array([x_min, y_min, t_min + eps, p_min]) + np.array([0.0, delta - y_min, t_max, p_max]) * lhs(4, int(N_b / 5))

    right_symmetry_s = np.array([delta, t_min + eps, p_min]) + np.array([y_max - delta, t_max, p_max]) * lhs(3, int(N_b * 2))
    right_symmetry_f = np.array([y_min, t_min + eps, p_min]) + np.array([delta - y_min, t_max, p_max]) * lhs(3, int(N_b / 3))
    xrss = right_symmetry_s[:, 2] * (P[1] - P[0]) + P[0]
    xrsf = right_symmetry_f[:, 2] * (P[1] - P[0]) + P[0]

    right_symmetry_s = np.concatenate((np.array([xrss]).T, right_symmetry_s), axis=1)
    right_symmetry_f = np.concatenate((np.array([xrsf]).T, right_symmetry_f), axis=1)

    bottom_wall = np.array([x_min, y_min, t_min + eps, p_min]) + np.array([x_max, 0.0, t_max, p_max]) * lhs(4, 3 * N_b)
    bottom_wall = delete(bottom_wall, N_b)
    top_wall = np.array([x_min, y_max, t_min + eps, p_min]) + np.array([x_max, 0.0, t_max, p_max]) * lhs(4, 3 * N_b)
    top_wall = delete(top_wall, N_b)
    middle_wall = np.array([x_min, delta, t_min + eps, p_min]) + np.array([x_max, 0.0, t_max, p_max]) * lhs(4, 3 * N_b)
    middle_wall = delete(middle_wall, N_b)

    xyt_bnd_left_s = np.array(left_wall_s)
    xyt_bnd_left_f = np.array(left_wall_f)
    xyt_bnd_right_s = np.array(right_symmetry_s)
    xyt_bnd_right_f = np.array(right_symmetry_f)
    xyt_bnd_bottom = np.array(bottom_wall)
    xyt_bnd_top = np.array(top_wall)
    xyt_bnd_middle = np.array(middle_wall)

    # Generate initial points
    xyt_ic_s = np.array([x_min, delta, t_min, p_min]) + np.array([x_max, y_max - delta, 0.0, p_max]) * lhs(4, 3 * N_i)
    xyt_ic_s = delete(xyt_ic_s, N_i)
    xyt_ic_f = np.array([x_min, y_min, t_min, p_min]) + np.array([x_max , delta - y_min, 0.0, p_max]) * lhs(4, 3 * N_i)
    xyt_ic_f = delete(xyt_ic_f, N_i)
    xyt_ic_S = np.array([x_min, t_min, p_min]) + np.array([x_max, 0.0, p_max]) * lhs(3, 3 * N_i)
    xyt_ic_S = delete(xyt_ic_S, N_i)

    # Generate collocation points
    xyt_col_s = mb_s + (ub - mb_s) * lhs(4, 3 * N_c_s)
    xyt_col_s = delete(xyt_col_s, N_c_s)
    xyt_col_f = lb + (mb_f - lb) * lhs(4, 3 * N_c_f)
    xyt_col_f = delete(xyt_col_f, N_c_f)
    xyt_col_S = np.concatenate((np.array([xyt_col_s[:,0]]).T, np.array([xyt_col_s[:,2]]).T, np.array([xyt_col_s[:,3]]).T), axis=1)

    # Convert to tensors
    domain_points = {
        'xyt_ic_s': torch.tensor(xyt_ic_s, dtype=torch.float32).to(device),
        'xyt_ic_f': torch.tensor(xyt_ic_f, dtype=torch.float32).to(device),
        'xyt_ic_S': torch.tensor(xyt_ic_S, dtype=torch.float32).to(device),
        'xyt_col_s': torch.tensor(xyt_col_s, dtype=torch.float32).to(device),
        'xyt_col_f': torch.tensor(xyt_col_f, dtype=torch.float32).to(device),
        'xyt_col_S': torch.tensor(xyt_col_S, dtype=torch.float32).to(device),
        'xyt_bnd_left_s': torch.tensor(xyt_bnd_left_s, dtype=torch.float32).to(device),
        'xyt_bnd_left_f': torch.tensor(xyt_bnd_left_f, dtype=torch.float32).to(device),
        'xyt_bnd_right_s': torch.tensor(xyt_bnd_right_s, dtype=torch.float32).to(device),
        'xyt_bnd_right_f': torch.tensor(xyt_bnd_right_f, dtype=torch.float32).to(device),
        'xyt_bnd_bottom': torch.tensor(xyt_bnd_bottom, dtype=torch.float32).to(device),
        'xyt_bnd_top': torch.tensor(xyt_bnd_top, dtype=torch.float32).to(device),
        'xyt_bnd_middle': torch.tensor(xyt_bnd_middle, dtype=torch.float32).to(device)
    }
    
    return domain_points