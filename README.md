Here's a comprehensive README file for your PDDL folder:

# PDDL Melting Simulation

This project implements a Physics-Informed Deep Learning (PDDL) approach for simulating melting processes with phase change interfaces.

## Project Structure

```
pddl/
├── config.py          # Configuration parameters and constants
├── domain.py          # Domain setup and point generation
├── networks.py        # Neural network architectures
├── training.py        # Training procedures and loss functions
├── utils.py           # Visualization and utility functions
├── main.py            # Main training script
└── README.md          # This file
```

## File Descriptions

### config.py
Contains all configuration parameters:
- Domain boundaries and physical constants
- Material properties (thermal conductivities, densities, etc.)
- Dimensionless numbers (Biot, Jakob numbers)
- Training parameters (point counts, optimization settings)
- Device configuration (CPU/GPU)

### domain.py
Handles domain point generation:
- Boundary point generation (walls, symmetry conditions)
- Initial condition points
- Collocation points for PDE residuals
- Geometric filtering for valid points

### networks.py
Defines neural network architectures:
- `UnifiedDNN`: Base neural network class
- `DNN1`: Solid temperature prediction network
- `DNN2`: Interface prediction network (sigmoid output)
- `DNN3`: Fluid temperature prediction network

### training.py
Core training logic:
- `AdaptiveWeightedPDDL`: Main training class with adaptive loss weighting
- `enhanced_adaptive_sampler`: Adaptive sampling for interface refinement
- Loss functions for BCs, ICs, interface conditions, and PDEs
- Optimization procedures (Adam + L-BFGS)

### utils.py
Visualization and analysis:
- Loss plotting functions
- Weight evolution tracking
- Result visualization utilities

### main.py
Main training script that orchestrates the entire training process.


## Training Process

The training follows a three-phase approach:

### Phase 0: Pretraining
- Uses provided CSV data to pretrain the networks
- Helps initialize weights with reasonable approximations
- Configurable via `PRE_EPS` in `config.py`

### Phase 1: Adam Optimization
- Adaptive sampling around the interface
- Learning rate scheduling
- Configurable via `ADAM_EPS` in `config.py`

### Phase 2: L-BFGS Refinement
- Multiple cycles of L-BFGS optimization
- Convergence monitoring
- Best model selection
- Configurable via `CY_LBFGS` and `IT_LBFGS` in `config.py`

## Key Features

### Adaptive Loss Weighting
- Automatically balances different loss components
- Prevents any single loss term from dominating
- Weights are updated during training based on loss evolution

### Enhanced Adaptive Sampling
- Curvature-based sampling around the interface
- Focuses computational resources on critical regions
- Improves interface prediction accuracy

### Multi-Network Architecture
- Separate networks for different physical quantities
- Specialized architectures for different prediction tasks
- Shared optimization for coupled physics

## Outputs

The training generates:
- Real-time loss monitoring in console
- Comprehensive loss plots (`comprehensive_loss.png`)
- Weight evolution plots (`weight_evolution.png`)
- Trained model parameters (in memory)

## Citation

If you use this code in your research, please cite the original work that this implementation is based on.
