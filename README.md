# Solidification Process Simulation with PDDL

A Physics-Driven Deep Learning (PDDL) approach for simulating solidification processes with moving phase change interfaces.

## 📊 Visualization Gallery

The simulation captures the solidification process across different aspect ratios:

| Aspect Ratio 1 | Aspect Ratio 2 | Aspect Ratio 3 |
|----------------|----------------|----------------|
| ![AR1](/animations/solidification_animation-2.gif) | ![AR2](results/ar2_solidification.gif) | ![AR3](results/ar3_solidification.gif) |

| Aspect Ratio 4 | Aspect Ratio 5 |
|----------------|----------------|
| ![AR4](results/ar4_solidification.gif) | ![AR5](results/ar5_solidification.gif) |

## 🧠 What is This Project?

This code implements a advanced neural network framework that solves the coupled heat transfer equations during solidification, where a liquid transforms into solid over time. The unique aspect is that it **learns the physics** without needing extensive simulation data - it uses the fundamental governing equations as constraints.

## 🔬 Key Features

- **Multi-Aspect Ratio Analysis**: Study solidification across 5 different geometric configurations (parameter P = 1-5)
- **Moving Interface Tracking**: Automatically predicts the solid-liquid boundary evolution
- **Adaptive Physics Learning**: Uses PDE constraints to ensure physical consistency
- **Efficient Neural Solvers**: Replaces traditional numerical methods with deep learning

## 📈 What the GIFs Show

Each animation demonstrates:
- **Temperature evolution** from initial to final state
- **Interface progression** as solidification advances
- **Geometric effects** of different aspect ratios on solidification patterns
- **Thermal boundary layers** development

## 🎯 Applications

- **Metal casting** and solidification processes
- **Crystal growth** in materials science
- **Phase change materials** for energy storage
- **Geological solidification** processes

## 🔍 Insights from Multi-Aspect Ratio Study

The 5 different aspect ratios (parameter P) reveal how geometry affects:
- **Solidification rate** and completion time
- **Interface curvature** and morphology
- **Temperature gradient** distributions
- **Heat extraction** efficiency

## 📚 Citation

If you use this code in your research, please cite the relevant foundational papers in physics-informed neural networks and solidification modeling.
