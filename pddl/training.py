"""
Training procedures and loss functions for PDDL
"""

import torch
import torch.nn as nn
from torch.autograd import grad
import time
import pandas as pd
from networks import *
from config import *
from domain import generate_domain_points

def enhanced_adaptive_sampler(Ss, xyt_col_s, xyt_col_S, xyt_col_f, n_extra=500, sigma=0.01, method='curvature'):
    """Enhanced adaptive sampling for interface refinement"""
    device = xyt_col_s.device
    S = Ss.squeeze()
    
    # Sort by x-coordinate for interface sampling
    x_sorted, idx = torch.sort(xyt_col_s[:,0])
    y_sorted = xyt_col_s[:,1][idx]
    t_sorted = xyt_col_s[:,2][idx]
    param_sorted = xyt_col_s[:,3][idx]
    S_sorted = S[idx]
    
    # Calculate normal vectors for interface
    dx = torch.gradient(x_sorted)[0]
    dS = torch.gradient(S_sorted)[0]
    dS[dS == 0] = 1e-6
    lengths = torch.sqrt(dx**2 + dS**2)
    nx = -dS / lengths
    ny = dx / lengths
    
    # Enhanced sampling strategies
    if method == 'curvature':
        # Curvature-based sampling
        d2S = torch.gradient(dS)[0]
        curvature = torch.abs(d2S) / (1 + dS**2)**1.5
        weights = curvature / (curvature.sum() + 1e-8)
        indices = torch.multinomial(weights, n_extra, replacement=True)
    elif method == 'gradient':
        # Gradient-based sampling
        weights = torch.abs(dS) / (torch.abs(dS).sum() + 1e-8)
        indices = torch.multinomial(weights, n_extra, replacement=True)
    else:
        # Uniform sampling
        indices = torch.randint(0, len(x_sorted), (n_extra,))
    
    xi = x_sorted[indices]
    yi = y_sorted[indices]
    ti = t_sorted[indices]
    pi = param_sorted[indices]
    nxi = nx[indices]
    nyi = ny[indices]
    
    offsets = torch.randn(n_extra, device=device) * sigma
    x_extra = xi + offsets * nxi
    y_extra = yi + offsets * nyi
    t_extra = ti
    p_extra = pi
    
    # Create enhanced collocation points
    extra_points = torch.stack([x_extra, y_extra, t_extra, p_extra], dim=1)
    new_xyt_col_s = torch.cat([xyt_col_s, extra_points], dim=0)
    
    # Pre-process interface points for interface_loss
    col_S_processed = xyt_col_S.clone()
    col_S_processed.requires_grad = True
    
    s = Ss  # Use the predicted S values
    
    # Create the modified X for interface calculations
    X_interface = torch.cat([xyt_col_s[:, 0:1], s, xyt_col_s[:, 2:3], xyt_col_s[:, 3:4]], dim=1)
    X_interface.requires_grad = True
    
    # Pre-process domain separation for pde_loss with smoother transition
    eps = 0.001
    solid_mask = (s >= xyt_col_s[:, 1:2] + eps).squeeze()
    liquid_mask = (s < xyt_col_s[:, 1:2] - eps).squeeze()
    
    col_s_solid = xyt_col_s[solid_mask].clone()
    col_s_liquid = xyt_col_s[liquid_mask].clone()
    col_f_fluid = xyt_col_f.clone()
    
    col_s_solid.requires_grad = True
    col_s_liquid.requires_grad = True
    col_f_fluid.requires_grad = True
    
    return {
        'col_s': new_xyt_col_s,
        'col_S_processed': col_S_processed,
        'X_interface': X_interface,
        'col_s_solid': col_s_solid,
        'col_s_liquid': col_s_liquid,
        'col_f_fluid': col_f_fluid
    }

class AdaptiveWeightedPDDL:
    def __init__(self, domain_points):
        self.domain_points = domain_points
        self.net1 = DNN1(dim_in=4, dim_out=3, n_layer=4, n_node=40, ub=ub, lb=lb).to(device)
        self.net2 = DNN2(dim_in=3, dim_out=1, n_layer=3, n_node=40, ub=ub, lb=lb).to(device)
        self.net3 = DNN3(dim_in=4, dim_out=3, n_layer=4, n_node=40, ub=ub, lb=lb).to(device)

        self.lbfgs = torch.optim.LBFGS(
            list(self.net1.parameters()) + list(self.net2.parameters()) + list(self.net3.parameters()),
            lr=1.0,
            max_iter=LBFGS_STP,
            max_eval=LBFGS_STP,
            tolerance_grad=1e-9,
            tolerance_change=1.0*np.finfo(float).eps,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        self.adam = torch.optim.Adam(
            list(self.net1.parameters()) + list(self.net2.parameters()) + list(self.net3.parameters()), 
            lr=1e-3
        )
        
        # Adaptive loss weights
        self.loss_weights = {
            'bc': nn.Parameter(torch.tensor(1.0)),
            'ic': nn.Parameter(torch.tensor(1.0)),
            'int': nn.Parameter(torch.tensor(1.0)),
            'pde': nn.Parameter(torch.tensor(1.0))
        }
        
        self.losses = {"bc": [], "ic": [], "int": [], "pde": []}
        self.weight_history = {key: [] for key in self.loss_weights.keys()}
        self.iter = 0
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.adam, step_size=10000, gamma=0.5)

    def predict_Ts(self, x):
        out1 = self.net1(x)
        Ts = out1[:, 0:1]
        qs_x = out1[:, 1:2]
        qs_y = out1[:, 2:3]
        return Ts, qs_x, qs_y

    def predict_s(self, x):
        out2 = self.net2(x)
        s = out2[:, 0:1]
        return s

    def predict_Tf(self, x):
        out3 = self.net3(x)
        Tf = out3[:, 0:1]
        qf_x = out3[:, 1:2]
        qf_y = out3[:, 2:3]
        return Tf, qf_x, qf_y

    def bc_loss(self):
        dp = self.domain_points
        Tf_left = self.predict_Tf(dp['xyt_bnd_left_f'])[0]
        qf_x_left = self.predict_Tf(dp['xyt_bnd_left_f'])[1]
        Tf_mid = self.predict_Tf(dp['xyt_bnd_middle'])[0]

        Ts_left = self.predict_Ts(dp['xyt_bnd_left_s'])[0]
        qs_x_left = self.predict_Ts(dp['xyt_bnd_left_s'])[1]
        Ts_mid = self.predict_Ts(dp['xyt_bnd_middle'])[0]

        qf_y_down = self.predict_Tf(dp['xyt_bnd_bottom'])[2]
        qf_x_right = self.predict_Tf(dp['xyt_bnd_right_f'])[1]
        qs_y_up = self.predict_Ts(dp['xyt_bnd_top'])[2]
        qs_x_right = self.predict_Ts(dp['xyt_bnd_right_s'])[1]

        mse_bc = (torch.mean(torch.square(Ts_left * Bi_s - qs_x_left)) + 
                 torch.mean(torch.square(Tf_left * Bi_f - qf_x_left)) + 
                 torch.mean(torch.square(Ts_mid - Tf_mid)) + 
                 torch.mean(torch.square(qf_y_down)) + 
                 torch.mean(torch.square(qf_x_right)) + 
                 torch.mean(torch.square(qs_y_up)) + 
                 torch.mean(torch.square(qs_x_right)))

        return mse_bc

    def ic_loss(self):
        dp = self.domain_points
        Ts = self.predict_Ts(dp['xyt_ic_s'])[0]
        S = self.predict_s(dp['xyt_ic_S'])
        Tf = self.predict_Tf(dp['xyt_ic_f'])[0]

        mse_ic_Ts = torch.mean(torch.square(Ts - T_initial))
        mse_ic_S = torch.mean(torch.square(S - delta))
        mse_ic_Tf = torch.mean(torch.square(Tf- T_initial))

        mse_ic = mse_ic_Ts + mse_ic_Tf + mse_ic_S 
        return mse_ic

    def interface_loss(self, X_interface, col_S_processed):
        # Ensure proper gradient tracking
        X_interface = X_interface.detach().requires_grad_(True)
        col_S_processed = col_S_processed.detach().requires_grad_(True)
        
        s = self.predict_s(col_S_processed)
        Ts = self.predict_Ts(X_interface)[2]

        ds_dx = grad(s.sum(), col_S_processed, create_graph=True)[0][:, 0:1]
        ds_dt = grad(s.sum(), col_S_processed, create_graph=True)[0][:, 1:2]
        dTs_dy = grad(Ts.sum(), X_interface, create_graph=True)[0][:, 1:2]

        pde_s = (1 + (ds_dx**2)) * (dTs_dy) - (1 / Ja) * ds_dt
        mse_pde_s = torch.mean(torch.square(pde_s))
        mse_temp = torch.mean(torch.square(Ts - T_m))

        mse_interface = mse_pde_s + mse_temp
        return mse_interface

    def pde_loss(self, col_s_solid, col_s_liquid, col_f_fluid):
        # Ensure proper gradient tracking
        col_s_solid = col_s_solid.detach().requires_grad_(True)
        col_s_liquid = col_s_liquid.detach().requires_grad_(True)
        col_f_fluid = col_f_fluid.detach().requires_grad_(True)
        
        Ts = self.predict_Ts(col_s_solid)[0]
        qs_x = self.predict_Ts(col_s_solid)[1]
        qs_y = self.predict_Ts(col_s_solid)[2]

        Tl = self.predict_Ts(col_s_liquid)[0]
        Tf = self.predict_Tf(col_f_fluid)[0]
        qf_x = self.predict_Tf(col_f_fluid)[1]
        qf_y = self.predict_Tf(col_f_fluid)[2]

        grads_Ts = grad(Ts.sum(), col_s_solid, create_graph=True)[0]
        dTs_dx = grads_Ts[:, 0:1]
        dTs_dy = grads_Ts[:, 1:2]
        dTs_dt = grads_Ts[:, 2:3]

        dTs_dxx = grad(qs_x.sum(), col_s_solid, create_graph=True)[0][:, 0:1]
        dTs_dyy = grad(qs_y.sum(), col_s_solid, create_graph=True)[0][:, 1:2]

        grads_Tf = grad(Tf.sum(), col_f_fluid, create_graph=True)[0]
        dTf_dx = grads_Tf[:, 0:1]
        dTf_dy = grads_Tf[:, 1:2]
        dTf_dt = grads_Tf[:, 2:3]

        dTf_dxx = grad(qf_x.sum(), col_f_fluid, create_graph=True)[0][:, 0:1]
        dTf_dyy = grad(qf_y.sum(), col_f_fluid, create_graph=True)[0][:, 1:2]

        pde_Ts = (dTs_dt) - (dTs_dxx + dTs_dyy)
        pde_Tf = alfa * (dTf_dt) - (dTf_dxx + dTf_dyy)

        mse_Tl = torch.mean(torch.square(Tl - T_m))
        mse_pde_Ts = torch.mean(torch.square(pde_Ts))
        mse_pde_Tf = torch.mean(torch.square(pde_Tf))

        mse_realization = (torch.mean(torch.square(qs_x - dTs_dx)) + 
                          torch.mean(torch.square(qs_y - dTs_dy)) + 
                          torch.mean(torch.square(qf_x - dTf_dx)) + 
                          torch.mean(torch.square(qf_y - dTf_dy)))

        mse_pde = mse_Tl + mse_pde_Ts + mse_pde_Tf + mse_realization
        return mse_pde

    def get_current_lr(self):
        return self.adam.param_groups[0]['lr']

    def update_loss_weights(self, losses_dict):
        tau=0.8
        with torch.no_grad():
            for key in self.loss_weights:
                current_loss = losses_dict[key]
                if len(self.losses[key]) > 1:
                    previous_loss = self.losses[key][-2]
                    ratio = current_loss / (previous_loss + 1e-8)
                    self.loss_weights[key].data = tau * self.loss_weights[key] + (1-tau) * ratio

    def closure(self, processed_points=None):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        
        if processed_points is None:
            with torch.no_grad():
                Ss = self.predict_s(self.domain_points['xyt_col_S'])
                processed_points = enhanced_adaptive_sampler(Ss, self.domain_points['xyt_col_s'], 
                                                           self.domain_points['xyt_col_S'], self.domain_points['xyt_col_f'])
        
        mse_bc = self.bc_loss()
        mse_ic = self.ic_loss()
        mse_int = self.interface_loss(processed_points['X_interface'], 
                                    processed_points['col_S_processed'])
        mse_pde = self.pde_loss(processed_points['col_s_solid'], 
                              processed_points['col_s_liquid'], 
                              processed_points['col_f_fluid'])

        weight_sum = 0

        weighted_loss = (torch.exp(-self.loss_weights['bc']) * mse_bc +
                        torch.exp(-self.loss_weights['ic']) * mse_ic +
                        torch.exp(-self.loss_weights['int']) * mse_int +
                        torch.exp(-self.loss_weights['pde']) * mse_pde +
                        weight_sum)  # Regularization

        weighted_loss.backward()

        # Store losses and weights
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["ic"].append(mse_ic.detach().cpu().item())
        self.losses["int"].append(mse_int.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        
        for key in self.loss_weights:
            self.weight_history[key].append(torch.exp(-self.loss_weights[key]).detach().cpu().item())

        current_losses = {
            'bc': mse_bc.detach(),
            'ic': mse_ic.detach(), 
            'int': mse_int.detach(),
            'pde': mse_pde.detach()
        }

        self.update_loss_weights(current_losses)

        self.iter += 1

        current_lr = self.get_current_lr()
        print(f"\r It: {self.iter} Loss: {weighted_loss.item():.5e} "
              f"BC: {mse_bc.item():.3e} IC: {mse_ic.item():.3e} "
              f"INT: {mse_int.item():.3e} PDE: {mse_pde.item():.3e} "
              f"LR: {current_lr:.2e}", end="")

        if self.iter % 500 == 0:
            print("")

        return weighted_loss

def pretrain_network(pddl, epochs, csv_file_path):
    """Pretrain the network using provided data"""
    data = pd.read_csv(csv_file_path)
    
    x_data = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
    t_data = torch.tensor(data['t'].values, dtype=torch.float32).unsqueeze(1).to(device)
    s_target_data = torch.tensor(data['s'].values, dtype=torch.float32).unsqueeze(1).to(device)
    Tf_target_data = torch.tensor(data['Tf'].values, dtype=torch.float32).unsqueeze(1).to(device)
    
    optimizer = torch.optim.Adam(
        list(pddl.net1.parameters()) + list(pddl.net2.parameters()) + list(pddl.net3.parameters()), 
        lr=5e-4
    )
    
    iter = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        s_pred = pddl.predict_s(torch.cat([x_data, t_data], dim=1))
        
        Tf_input = torch.cat([x_data, torch.zeros_like(x_data), t_data, torch.zeros_like(x_data)], dim=1)
        Tf_pred = pddl.predict_Tf(Tf_input)[0]
        
        s_loss = torch.mean(torch.square(s_target_data - s_pred))
        Tf_loss = torch.mean(torch.square(Tf_target_data - Tf_pred))
        
        loss = s_loss + Tf_loss
        
        loss.backward()
        optimizer.step()
        
        print(f"\r Pretrain It: {iter} Loss: {loss.item():.5e} s_Loss: {s_loss.item():.5e} Tf_Loss: {Tf_loss.item():.5e}", end="")
        iter += 1
    
    print("\nPretraining completed!")

def adaptive_lbfgs_training(pddl, max_cycles=10, lbfgs_steps_per_cycle=5):
    """Adaptive L-BFGS training with convergence monitoring"""
    previous_Ss = None
    best_cycle_loss = float('inf')
    best_cycle_state = None
    
    for cycle in range(max_cycles):
        print(f"\n--- Adaptive L-BFGS Cycle {cycle + 1}/{max_cycles} ---")
        
        with torch.no_grad():
            Ss = pddl.predict_s(pddl.domain_points['xyt_col_S'])
            current_points = enhanced_adaptive_sampler(Ss, pddl.domain_points['xyt_col_s'], 
                                                     pddl.domain_points['xyt_col_S'], pddl.domain_points['xyt_col_f'])
        
        if previous_Ss is not None:
            interface_change = torch.mean(torch.abs(Ss - previous_Ss)).item()
            print(f"Interface change from previous cycle: {interface_change:.3e}")
        
        previous_loss = float('inf')
        cycle_converged = False
        
        for step in range(lbfgs_steps_per_cycle):
            def lbfgs_closure():
                return pddl.closure(current_points)
            
            current_loss = pddl.lbfgs.step(lbfgs_closure)
            
            if step > 0 and abs(current_loss - previous_loss) < 1e-8:
                print(f"  Cycle {cycle + 1} converged at step {step + 1}")
                cycle_converged = True
                break
            
            previous_loss = current_loss
            
            if current_loss < best_cycle_loss:
                best_cycle_loss = current_loss
                best_cycle_state = {
                    'net1': pddl.net1.state_dict(),
                    'net2': pddl.net2.state_dict(), 
                    'net3': pddl.net3.state_dict(),
                    'cycle': cycle,
                    'step': step,
                    'loss': best_cycle_loss
                }
        
        print(f"Completed L-BFGS cycle {cycle + 1}")
        
        previous_Ss = Ss.clone()
        
        if cycle > 1 and interface_change < 1e-5:
            print(f"Global convergence: interface stabilized after {cycle + 1} cycles")
            break
            
        if cycle_converged and cycle > 2:
            print(f"Global convergence: loss stabilized after {cycle + 1} cycles")
            break
    
    if best_cycle_state:
        pddl.net1.load_state_dict(best_cycle_state['net1'])
        pddl.net2.load_state_dict(best_cycle_state['net2'])
        pddl.net3.load_state_dict(best_cycle_state['net3'])
        print(f"Loaded best L-BFGS model from cycle {best_cycle_state['cycle'] + 1}, step {best_cycle_state['step'] + 1}")
        print(f"Best L-BFGS loss: {best_cycle_state['loss']:.3e}")
    
    return pddl