"""
Main script for PDDL solidification simulation
"""

import time
from training import AdaptiveWeightedPDDL, pretrain_network, adaptive_lbfgs_training
from domain import generate_domain_points
from utils import plot_comprehensive_loss, plot_weight_history
from config import *

def main():
    print(f"Using device: {device}")
    
    # Generate domain points
    print("Generating domain points...")
    domain_points = generate_domain_points()
    
    # Initialize PDDL model
    print("Initializing PDDL model...")
    pddl = AdaptiveWeightedPDDL(domain_points)
    
    start_time = time.time()
    
    # Pretraining phase
    print("\n=== Phase 1: Pretraining ===")
    pretrain_network(pddl, PRE_EPS, "path/to/your/paper_data.csv")
    
    # Adam optimization phase
    print("\n=== Phase 2: Adam with adaptive sampling ===")
    for i in range(ADAM_EPS):
        with torch.no_grad():
            Ss = pddl.predict_s(domain_points['xyt_col_S'])
            processed_points = enhanced_adaptive_sampler(Ss, domain_points['xyt_col_s'], 
                                                       domain_points['xyt_col_S'], domain_points['xyt_col_f'])
        
        pddl.closure(processed_points)
        pddl.adam.step()
        pddl.scheduler.step()
        
    # L-BFGS optimization phase
    print("\n=== Phase 3: Adaptive L-BFGS ===")
    pddl = adaptive_lbfgs_training(pddl, max_cycles=CY_LBFGS, lbfgs_steps_per_cycle=IT_LBFGS)
    
    # Final refinement
    print("\n=== Final Phase: L-BFGS refinement ===")
    with torch.no_grad():
        Ss_final = pddl.predict_s(domain_points['xyt_col_S'])
        final_points = enhanced_adaptive_sampler(Ss_final, domain_points['xyt_col_s'], 
                                               domain_points['xyt_col_S'], domain_points['xyt_col_f'])
    
    def final_lbfgs_closure():
        return pddl.closure(final_points)
    
    pddl.lbfgs.step(final_lbfgs_closure)
    
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Plot results
    print("\nGenerating plots...")
    plot_comprehensive_loss(pddl.losses, "comprehensive_loss.png")
    plot_weight_history(pddl.weight_history, "weight_evolution.png")
    
    print("All done!")

if __name__ == "__main__":
    main()