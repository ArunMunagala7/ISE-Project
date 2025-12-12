"""
Quick test of Q-Learning integration
"""
import numpy as np
from config.params import *
from src.qlearning_controller import QLearningController

# Load model
controller = QLearningController(
    num_bins=QL_NUM_BINS,
    num_actions=QL_NUM_ACTIONS,
    alpha=QL_ALPHA,
    gamma=QL_GAMMA,
    epsilon=0.0
)

import os
if os.path.exists(QL_MODEL_PATH):
    controller.load_model(QL_MODEL_PATH)
    print(f"✓ Loaded Q-Learning model from {QL_MODEL_PATH}")
    print(f"  Q-table shape: {controller.q_table.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(controller.q_table)}/{controller.q_table.size}")
    print(f"  Q-value range: [{controller.q_table.min():.3f}, {controller.q_table.max():.3f}]")
    print(f"  Mean Q-value: {controller.q_table[controller.q_table != 0].mean():.3f}")
    
    # Test a few control decisions
    print("\n✓ Testing learned control:")
    test_states = [
        (0.5, 0.1, "Small lateral, small heading error"),
        (2.0, 0.3, "Large lateral, medium heading error"),
        (0.1, 0.5, "Small lateral, large heading error"),
    ]
    
    for lat_err, head_err, desc in test_states:
        offset, state, action = controller.get_control_offset(lat_err, head_err, training=False)
        print(f"  {desc}:")
        print(f"    Lateral={lat_err:.2f}m, Heading={np.degrees(head_err):.1f}°")
        print(f"    → Action offset: {offset:.3f} rad/s")
else:
    print(f"✗ No model found at {QL_MODEL_PATH}")
