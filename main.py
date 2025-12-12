"""
Main SLAM Simulation Script
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Import project modules
from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.visualization import SLAMVisualizer
from src.utils import generate_random_landmarks


def run_slam_simulation(args):
    """
    Run complete SLAM simulation
    
    Args:
        args: Command line arguments
    """
    print("="*60)
    print("SLAM SIMULATION - Extended Kalman Filter")
    print("="*60)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Generate environment
    print(f"\n--- Environment Setup ---")
    true_landmarks = generate_random_landmarks(args.num_landmarks, LANDMARK_AREA_SIZE)
    print(f"Generated {args.num_landmarks} landmarks")
    print(f"Sensor range: {MAX_RANGE}m, FOV: {np.degrees(FOV_ANGLE)}°")
    
    # Initialize robot
    print(f"\n--- Robot Initialization ---")
    robot = Robot(INITIAL_STATE, DT)
    print(f"Initial state: x={INITIAL_STATE[0]:.2f}, y={INITIAL_STATE[1]:.2f}, θ={INITIAL_STATE[2]:.2f}")
    
    # Initialize Q-Learning controller if needed
    qlearning_controller = None
    if CONTROL_TYPE == "qlearning":
        from src.qlearning_controller import QLearningController
        qlearning_controller = QLearningController(
            num_bins=QL_NUM_BINS,
            num_actions=QL_NUM_ACTIONS,
            alpha=QL_ALPHA,
            gamma=QL_GAMMA,
            epsilon=QL_EPSILON
        )
        # Try to load existing model
        if os.path.exists(QL_MODEL_PATH):
            qlearning_controller.load_model(QL_MODEL_PATH)
            print(f"Loaded Q-Learning model from {QL_MODEL_PATH}")
        print(f"Q-Learning control enabled (training={QL_TRAINING})")
    
    # Initialize trajectory controller
    controller_params = {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': 5.0
    }
    controller = TrajectoryController(
        args.trajectory, 
        controller_params,
        control_type=CONTROL_TYPE,
        qlearning_controller=qlearning_controller
    )
    print(f"Trajectory: {args.trajectory}, Control: {CONTROL_TYPE}")
    
    # Initialize EKF-SLAM
    print(f"\n--- EKF-SLAM Initialization ---")
    ekf_slam = EKF_SLAM(INITIAL_STATE, args.num_landmarks, INITIAL_STATE_COV)
    print(f"State dimension: {len(ekf_slam.mu)}")
    print(f"Robot state: 3, Landmarks: {2 * args.num_landmarks}")
    
    # Initialize data association
    data_assoc = DataAssociation(ekf_slam)
    
    # Initialize visualizer
    print(f"\n--- Visualization Setup ---")
    visualizer = SLAMVisualizer(true_landmarks, save_video=args.save_video)
    visualizer.true_trajectory = [robot.get_state()]
    
    # Simulation loop
    print(f"\n--- Starting Simulation ---")
    print(f"Duration: {SIM_TIME}s, Time step: {DT}s")
    
    num_steps = int(SIM_TIME / DT)
    t = 0.0
    
    plt.ion()  # Interactive mode for animation
    plt.show(block=False)
    
    print("\n" + "="*60)
    print("🎬 ANIMATION RUNNING - Watch the plot window!")
    print("   The plot will update live as the robot moves")
    print("   Look for: uncertainty shrinking, trajectories growing")
    print("="*60 + "\n")
    
    for step in range(num_steps):
        t = step * DT
        
        # Get current robot state
        robot_state = robot.get_state()
        
        # Compute control input
        v, w = controller.get_control(robot_state, t)
        
        # --- PREDICTION STEP ---
        # Robot moves (ground truth)
        v_actual, w_actual = robot.move(v, w, add_noise=True)
        
        # EKF predicts next state
        ekf_slam.predict(v, w, DT)
        
        # --- MEASUREMENT STEP ---
        # Simulate sensor measurements
        measurements, measured_landmark_ids = data_assoc.simulate_measurements(
            robot.get_state(), true_landmarks, add_noise=True
        )
        
        # Get visible landmarks for data association
        visible_ids = data_assoc.get_visible_landmarks(robot.get_state(), true_landmarks)
        
        # Associate measurements with landmarks
        if len(measurements) > 0:
            # For simplicity, we know the correspondence (measured_landmark_ids)
            # In real scenario, we'd use: associations = data_assoc.associate_measurements(measurements, visible_ids)
            # Here we use ground truth association
            for landmark_id, measurement in zip(measured_landmark_ids, measurements):
                ekf_slam.update(landmark_id, measurement)
        
        # --- VISUALIZATION ---
        if step % PLOT_INTERVAL == 0 or step == num_steps - 1:
            robot_est = ekf_slam.get_robot_state()
            visualizer.true_trajectory.append(robot.get_state())
            visualizer.update_plot(robot.get_state(), robot_est, ekf_slam, t)
            visualizer.update_error_plot(robot.get_state(), robot_est, ekf_slam, t)
            visualizer.show()
            
            # Print progress with animation indicator
            num_initialized = np.sum(ekf_slam.landmark_initialized)
            if step % (num_steps // 10) == 0:
                progress = int(50 * step / num_steps)
                bar = '█' * progress + '░' * (50 - progress)
                print(f"  [{bar}] {100*step//num_steps}% | "
                      f"t={t:.1f}s | Landmarks: {num_initialized}/{args.num_landmarks}")
    
    plt.ioff()
    
    # Final statistics
    print(f"\n--- Simulation Complete ---")
    robot_final_true = robot.get_state()
    robot_final_est = ekf_slam.get_robot_state()
    position_error = np.linalg.norm(robot_final_true[0:2] - robot_final_est[0:2])
    
    print(f"Final robot position error: {position_error:.3f}m")
    print(f"Landmarks initialized: {np.sum(ekf_slam.landmark_initialized)}/{args.num_landmarks}")
    
    # Compute average landmark error
    landmark_errors = []
    for i in range(args.num_landmarks):
        if ekf_slam.landmark_initialized[i]:
            est = ekf_slam.get_landmark_state(i)
            true = true_landmarks[i]
            error = np.linalg.norm(est - true)
            landmark_errors.append(error)
    
    if landmark_errors:
        avg_landmark_error = np.mean(landmark_errors)
        print(f"Average landmark position error: {avg_landmark_error:.3f}m")
    
    # Save Q-Learning model if used
    if qlearning_controller is not None and QL_TRAINING:
        os.makedirs(os.path.dirname(QL_MODEL_PATH), exist_ok=True)
        qlearning_controller.save_model(QL_MODEL_PATH)
        print(f"\n--- Q-Learning Statistics ---")
        print(f"Model saved to: {QL_MODEL_PATH}")
        q_values = qlearning_controller.q_table
        print(f"Q-table shape: {q_values.shape}")
        print(f"Non-zero entries: {np.count_nonzero(q_values)}/{q_values.size}")
        print(f"Q-value range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # Save plots
    print(f"\n--- Saving Results ---")
    visualizer.save_final_plots(args.output_dir)
    
    # Keep final plot open
    if not args.no_display:
        print("\nClose the plot window to exit...")
        plt.show()
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EKF-SLAM Simulation')
    
    # Simulation parameters
    parser.add_argument('--trajectory', type=str, default=TRAJECTORY_TYPE,
                       choices=['circle', 'figure8'],
                       help='Trajectory type')
    parser.add_argument('--num_landmarks', type=int, default=NUM_LANDMARKS,
                       help='Number of landmarks')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs/plots',
                       help='Output directory for plots')
    parser.add_argument('--save_video', action='store_true',
                       help='Save animation as video')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plots (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_video:
        os.makedirs('outputs/videos', exist_ok=True)
    
    # Run simulation
    run_slam_simulation(args)


if __name__ == "__main__":
    main()
