"""
Compare Feedback vs Q-Learning Control
Runs both controllers and compares performance metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks
from src.qlearning_controller import QLearningController


def run_simulation(control_type="feedback", qlearning_controller=None, seed=42):
    """
    Run SLAM simulation with specified control type
    
    Args:
        control_type: "feedback" or "qlearning"
        qlearning_controller: QLearningController instance (for Q-Learning)
        seed: Random seed
    
    Returns:
        dict with metrics
    """
    np.random.seed(seed)
    
    # Generate environment
    true_landmarks = generate_random_landmarks(NUM_LANDMARKS, LANDMARK_AREA_SIZE)
    robot = Robot(INITIAL_STATE, DT)
    
    # Initialize controller
    controller_params = {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': FIGURE8_SCALE
    }
    controller = TrajectoryController(
        TRAJECTORY_TYPE, 
        controller_params,
        control_type=control_type,
        qlearning_controller=qlearning_controller
    )
    
    # Initialize EKF-SLAM
    ekf_slam = EKF_SLAM(INITIAL_STATE, NUM_LANDMARKS, INITIAL_STATE_COV)
    data_assoc = DataAssociation(ekf_slam)
    
    # Storage
    true_trajectory = []
    estimated_trajectory = []
    lateral_errors = []
    heading_errors = []
    
    # Run simulation
    num_steps = int(SIM_TIME / DT)
    
    for step in range(num_steps):
        t = step * DT
        robot_state = robot.get_state()
        
        # Compute control
        v, w = controller.get_control(robot_state, t)
        
        # Move robot
        robot.move(v, w, add_noise=True)
        
        # EKF prediction
        ekf_slam.predict(v, w, DT)
        
        # Get measurements
        measurements, measured_landmark_ids = data_assoc.simulate_measurements(
            robot_state, true_landmarks, add_noise=True
        )
        
        # EKF update
        if len(measurements) > 0:
            for landmark_id, measurement in zip(measured_landmark_ids, measurements):
                ekf_slam.update(landmark_id, measurement)
        
        # Store data
        true_trajectory.append(robot_state.copy())
        estimated_trajectory.append(ekf_slam.get_robot_state().copy())
        
        # Compute tracking errors if Q-Learning
        if control_type == "qlearning" and qlearning_controller is not None:
            from src.qlearning_controller import compute_tracking_errors
            target = controller.trajectory_waypoints[min(step*10, len(controller.trajectory_waypoints)-1)]
            lat_err, head_err = compute_tracking_errors(
                robot_state, target, controller.trajectory_waypoints
            )
            lateral_errors.append(abs(lat_err))
            heading_errors.append(abs(head_err))
    
    # Compute metrics
    true_trajectory = np.array(true_trajectory)
    estimated_trajectory = np.array(estimated_trajectory)
    
    position_errors = np.linalg.norm(
        true_trajectory[:, :2] - estimated_trajectory[:, :2], axis=1
    )
    
    metrics = {
        'final_position_error': position_errors[-1],
        'avg_position_error': np.mean(position_errors),
        'max_position_error': np.max(position_errors),
        'true_trajectory': true_trajectory,
        'estimated_trajectory': estimated_trajectory,
        'position_errors': position_errors
    }
    
    if lateral_errors:
        metrics['avg_lateral_error'] = np.mean(lateral_errors)
        metrics['avg_heading_error'] = np.mean(heading_errors)
    
    # Count landmarks
    num_initialized = np.sum(ekf_slam.landmark_initialized)
    metrics['landmarks_found'] = num_initialized
    
    # Compute landmark errors
    landmark_errors = []
    for i in range(num_initialized):
        if ekf_slam.landmark_initialized[i]:
            est = ekf_slam.get_landmark_state(i)
            true = true_landmarks[i]
            error = np.linalg.norm(est - true)
            landmark_errors.append(error)
    
    if landmark_errors:
        metrics['avg_landmark_error'] = np.mean(landmark_errors)
    
    return metrics


def main():
    """Compare feedback vs Q-Learning control"""
    
    print("="*70)
    print("COMPARING FEEDBACK vs Q-LEARNING CONTROL")
    print("="*70)
    
    # Run feedback controller
    print("\n1. Running with FEEDBACK control...")
    feedback_metrics = run_simulation(control_type="feedback", seed=42)
    
    # Initialize Q-Learning controller
    print("\n2. Loading Q-Learning model...")
    qlearning_controller = QLearningController(
        num_bins=QL_NUM_BINS,
        num_actions=QL_NUM_ACTIONS,
        alpha=QL_ALPHA,
        gamma=QL_GAMMA,
        epsilon=0.0  # Pure exploitation
    )
    
    if os.path.exists(QL_MODEL_PATH):
        qlearning_controller.load_model(QL_MODEL_PATH)
        print(f"   ✓ Loaded from {QL_MODEL_PATH}")
    else:
        print(f"   ✗ No trained model found at {QL_MODEL_PATH}")
        print("   Using untrained Q-table (will perform poorly)")
    
    # Run Q-Learning controller
    print("\n3. Running with Q-LEARNING control (exploitation only)...")
    qlearning_metrics = run_simulation(
        control_type="qlearning", 
        qlearning_controller=qlearning_controller,
        seed=42
    )
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Feedback':<15} {'Q-Learning':<15} {'Winner':<10}")
    print("-"*70)
    
    # Position errors
    fb_final = feedback_metrics['final_position_error']
    ql_final = qlearning_metrics['final_position_error']
    winner = "Q-Learning" if ql_final < fb_final else "Feedback"
    print(f"{'Final Position Error (m)':<30} {fb_final:<15.3f} {ql_final:<15.3f} {winner:<10}")
    
    fb_avg = feedback_metrics['avg_position_error']
    ql_avg = qlearning_metrics['avg_position_error']
    winner = "Q-Learning" if ql_avg < fb_avg else "Feedback"
    print(f"{'Avg Position Error (m)':<30} {fb_avg:<15.3f} {ql_avg:<15.3f} {winner:<10}")
    
    # Landmark errors
    fb_lm = feedback_metrics.get('avg_landmark_error', 0)
    ql_lm = qlearning_metrics.get('avg_landmark_error', 0)
    winner = "Q-Learning" if ql_lm < fb_lm else "Feedback"
    print(f"{'Avg Landmark Error (m)':<30} {fb_lm:<15.3f} {ql_lm:<15.3f} {winner:<10}")
    
    # Landmarks found
    fb_found = feedback_metrics['landmarks_found']
    ql_found = qlearning_metrics['landmarks_found']
    winner = "Q-Learning" if ql_found > fb_found else "Feedback"
    print(f"{'Landmarks Found':<30} {fb_found:<15} {ql_found:<15} {winner:<10}")
    
    # Tracking errors (Q-Learning only)
    if 'avg_lateral_error' in qlearning_metrics:
        print(f"\n{'Q-Learning Tracking Performance:':<30}")
        print(f"  Avg Lateral Error: {qlearning_metrics['avg_lateral_error']:.3f}m")
        print(f"  Avg Heading Error: {np.degrees(qlearning_metrics['avg_heading_error']):.1f}°")
    
    # Create comparison plots
    print("\n4. Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trajectories
    ax = axes[0, 0]
    ax.plot(feedback_metrics['true_trajectory'][:, 0], 
            feedback_metrics['true_trajectory'][:, 1], 
            'k-', linewidth=2, label='True', alpha=0.5)
    ax.plot(feedback_metrics['estimated_trajectory'][:, 0], 
            feedback_metrics['estimated_trajectory'][:, 1], 
            'b--', linewidth=1.5, label='Feedback Est.')
    ax.plot(qlearning_metrics['estimated_trajectory'][:, 0], 
            qlearning_metrics['estimated_trajectory'][:, 1], 
            'r--', linewidth=1.5, label='Q-Learning Est.')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Estimated Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Position errors over time
    ax = axes[0, 1]
    time_steps = np.arange(len(feedback_metrics['position_errors'])) * DT
    ax.plot(time_steps, feedback_metrics['position_errors'], 'b-', label='Feedback', linewidth=1.5)
    ax.plot(time_steps, qlearning_metrics['position_errors'], 'r-', label='Q-Learning', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1, 0]
    ax.hist(feedback_metrics['position_errors'], bins=30, alpha=0.5, label='Feedback', color='blue')
    ax.hist(qlearning_metrics['position_errors'], bins=30, alpha=0.5, label='Q-Learning', color='red')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary bar chart
    ax = axes[1, 1]
    metrics_names = ['Final\nPos Error', 'Avg\nPos Error', 'Avg\nLandmark Error']
    fb_values = [fb_final, fb_avg, fb_lm]
    ql_values = [ql_final, ql_avg, ql_lm]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, fb_values, width, label='Feedback', color='blue', alpha=0.7)
    ax.bar(x + width/2, ql_values, width, label='Q-Learning', color='red', alpha=0.7)
    
    ax.set_ylabel('Error (m)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs/comparison', exist_ok=True)
    save_path = 'outputs/comparison/feedback_vs_qlearning.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    
    # Show plot
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
