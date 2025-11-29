"""
Configuration parameters for the PDDL solidification simulation
"""

import numpy as np
import torch

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Domain boundaries
Lc = 0.01
t_ref = 830 * 1920 * (0.01 ** 2) / 0.514

x_min, x_max = 0.0, 2.5
y_min, y_max = 0.0, 1.0
t_min, t_max = 0.0, 1000 / t_ref
p_min, p_max = 0.0, 1.0

P = [0.5, 2.5]

# Boundary temperatures
temp_inf = 10
temp_initial = 25

T_inf = 0.0
T_initial = 1.0
T_m = 1.0

delta = 0.0005 / Lc  # delta = 0.0005

# Material properties
rho_s = 830
c_s = 1920
c_l = 3260
k_s = 0.514
k_l = 0.224
rho_f = 2770
c_f = 875
k_f = 177

# Thermal diffusivities
a_f = k_f / (rho_f * c_f)
a_s = k_s / (rho_s * c_s)
alfa = a_f / a_s

# Dimensionless numbers
L = 251000
Ja = c_s * (temp_initial - temp_inf) / L  # 0.218

h = 65
Bi_s = h * Lc / k_s
Bi_f = h * Lc / k_f

# Training parameters
N_b = 4000
N_i = 10000
N_c_s = 60000
N_c_f = 10000
N_h = 2000

# Domain bounds
ub = np.array([x_max, y_max, t_max, p_max])
lb = np.array([x_min, y_min, t_min, p_min])
mb_f = np.array([x_max, delta, t_max, p_max])
mb_s = np.array([x_min, delta, t_min, p_min])

eps = 0.01

# Optimization parameters
LBFGS_STP = 100
ADAM_EPS = 1000
CY_LBFGS = 10
IT_LBFGS = 5
PRE_EPS = 5000