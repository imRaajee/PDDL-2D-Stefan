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

## Quick Start

1. **Install dependencies**:
```bash
pip install torch numpy matplotlib pandas scipy pyDOE
```

2. **Prepare data** (if using pretraining):
   - Place your CSV file with pretraining data in the appropriate location
   - Update the file path in `main.py` (line ~20 in pretrain_network call)

3. **Run training**:
```bash
python main.py
```

## Training Process

The training follows a three-phase approach:

### Phase 1: Pretraining
- Uses provided CSV data to pretrain the networks
- Helps initialize weights with reasonable approximations
- Configurable via `PRE_EPS` in `config.py`

### Phase 2: Adam Optimization
- Adaptive sampling around the interface
- Learning rate scheduling
- Configurable via `ADAM_EPS` in `config.py`

### Phase 3: L-BFGS Refinement
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

## Configuration

Modify `config.py` to adjust:

- **Physical parameters**: Material properties, boundary conditions
- **Domain settings**: Spatial and temporal boundaries
- **Training parameters**: Point counts, optimization settings
- **Network architecture**: Layers, nodes, activation functions

## Outputs

The training generates:
- Real-time loss monitoring in console
- Comprehensive loss plots (`comprehensive_loss.png`)
- Weight evolution plots (`weight_evolution.png`)
- Trained model parameters (in memory)

## Troubleshooting

### Common Issues

1. **Memory errors**:
   - Reduce `N_c_s`, `N_c_f`, `N_b`, `N_i` in `config.py`
   - Use smaller network architectures

2. **Training instability**:
   - Adjust learning rates in `training.py`
   - Modify adaptive weight parameters (tau value)
   - Check physical parameter consistency

3. **Slow convergence**:
   - Increase pretraining epochs (`PRE_EPS`)
   - Adjust L-BFGS cycle parameters
   - Verify boundary condition implementations

### Performance Tips

- Use GPU for faster training (automatically detected)
- Monitor interface change during L-BFGS cycles for convergence
- Adjust adaptive sampling parameters for your specific problem
- Consider domain decomposition for larger problems

## Extending the Code

To modify for different problems:

1. **New physics**: Update loss functions in `training.py`
2. **Different domains**: Modify point generation in `domain.py`
3. **Alternative architectures**: Extend networks in `networks.py`
4. **Additional constraints**: Add new loss terms to `AdaptiveWeightedPDDL`

## Citation

If you use this code in your research, please cite the original work that this implementation is based on.
