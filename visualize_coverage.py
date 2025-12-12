"""
Visualize Landmark Coverage
Shows which landmarks were observed during the simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from config.params import *
from src.ekf_slam import EKF_SLAM
from src.robot import Robot, TrajectoryController
from src.utils import generate_random_landmarks
from src.data_association import DataAssociation
import os


def run_simulation():
    """Run simulation and return coverage data"""
    # Initialize
    np.random.seed(42)
    landmarks = generate_random_landmarks(NUM_LANDMARKS, LANDMARK_AREA_SIZE)
    robot = Robot(INITIAL_STATE, DT)
    controller = TrajectoryController(TRAJECTORY_TYPE, {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': FIGURE8_SCALE
    })
    ekf = EKF_SLAM(INITIAL_STATE, NUM_LANDMARKS, INITIAL_STATE_COV)
    data_assoc = DataAssociation(ekf)
    
    # Tracking
    observation_counts = np.zeros(NUM_LANDMARKS, dtype=int)
    
    # Simulate
    num_steps = int(SIM_TIME / DT)
    
    for step in range(num_steps):
        t = step * DT
        robot_state = robot.get_state()
        
        # Robot motion
        v, w = controller.get_control(robot_state, t)
        robot.move(v, w, add_noise=True)
        
        # EKF prediction
        ekf.predict(v, w, DT)
        
        # Sensor measurements
        measurements, measured_landmark_ids = data_assoc.simulate_measurements(
            robot.get_state(), landmarks, add_noise=True
        )
        
        # Process measurements
        for landmark_id in measured_landmark_ids:
            observation_counts[landmark_id] += 1
    
    return landmarks, ekf, observation_counts


def visualize_coverage(landmarks, ekf, observation_counts):
    """Create coverage visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- LEFT PLOT: Spatial Coverage ---
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Landmark Coverage Map', fontsize=14, fontweight='bold')
    
    # Plot trajectory range
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Figure-8 path
    t_path = np.linspace(0, SIM_TIME, 500)
    omega = 0.15
    x_path = FIGURE8_SCALE * np.sin(omega * t_path)
    y_path = FIGURE8_SCALE * np.sin(omega * t_path) * np.cos(omega * t_path)
    ax1.plot(x_path, y_path, 'b-', alpha=0.3, linewidth=2, label='Robot Path')
    
    # Sensor range envelope (approximate)
    max_reach = FIGURE8_SCALE + MAX_RANGE
    circle_outer = Circle((0, 0), max_reach, fill=False, edgecolor='blue', 
                          linestyle='--', alpha=0.3, linewidth=1.5, 
                          label=f'Max Reach (~{max_reach:.1f}m)')
    ax1.add_patch(circle_outer)
    
    # Plot landmarks
    for i, lm in enumerate(landmarks):
        obs_count = observation_counts[i]
        
        if obs_count == 0:
            # Never observed - red X
            ax1.plot(lm[0], lm[1], 'rx', markersize=15, markeredgewidth=3,
                    label='Never Seen' if i == 16 else '')
            ax1.text(lm[0], lm[1]+0.5, f'#{i}', ha='center', fontsize=9,
                    color='red', fontweight='bold')
        else:
            # Observed - color by confidence
            lm_est = ekf.get_landmark_state(i)
            error = np.linalg.norm(lm - lm_est)
            
            # Color based on error
            if error < 0.5:
                color = 'green'
                marker = 'o'
                alpha = 0.8
            elif error < 1.0:
                color = 'yellow'
                marker = 'o'
                alpha = 0.7
            else:
                color = 'orange'
                marker = 'o'
                alpha = 0.6
            
            # True position
            ax1.plot(lm[0], lm[1], marker, color=color, markersize=10, 
                    alpha=alpha, markeredgecolor='black', markeredgewidth=1)
            
            # Estimated position (smaller)
            ax1.plot(lm_est[0], lm_est[1], 's', color=color, markersize=6,
                    alpha=alpha, markeredgecolor='black', markeredgewidth=1)
            
            # Connection line
            ax1.plot([lm[0], lm_est[0]], [lm[1], lm_est[1]], 'k--', 
                    alpha=0.3, linewidth=0.8)
            
            # Label
            ax1.text(lm[0], lm[1]+0.5, f'#{i}\n{obs_count}obs', ha='center', 
                    fontsize=8, fontweight='bold')
    
    # Legend for spatial plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Error < 0.5m (Excellent)'),
        Patch(facecolor='yellow', alpha=0.7, label='Error 0.5-1.0m (Good)'),
        Patch(facecolor='orange', alpha=0.6, label='Error > 1.0m (Fair)'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                   markersize=12, markeredgewidth=3, label='Never Observed'),
        plt.Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, 
                   label='Robot Path'),
        plt.Line2D([0], [0], color='blue', alpha=0.3, linewidth=1.5, 
                   linestyle='--', label=f'Max Reach (~{max_reach:.1f}m)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax1.set_xlim(-max_reach-2, max_reach+2)
    ax1.set_ylim(-max_reach-2, max_reach+2)
    
    # --- RIGHT PLOT: Observation Statistics ---
    observed_landmarks = np.where(observation_counts > 0)[0]
    unobserved_landmarks = np.where(observation_counts == 0)[0]
    
    # Bar chart of observations
    colors = []
    for i in range(NUM_LANDMARKS):
        if observation_counts[i] == 0:
            colors.append('red')
        else:
            lm_est = ekf.get_landmark_state(i)
            error = np.linalg.norm(landmarks[i] - lm_est)
            if error < 0.5:
                colors.append('green')
            elif error < 1.0:
                colors.append('yellow')
            else:
                colors.append('orange')
    
    bars = ax2.bar(range(NUM_LANDMARKS), observation_counts, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Landmark ID', fontsize=12)
    ax2.set_ylabel('Number of Observations', fontsize=12)
    ax2.set_title('Observation Counts per Landmark', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(NUM_LANDMARKS))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add text summary
    coverage_pct = (len(observed_landmarks) / NUM_LANDMARKS) * 100
    avg_obs = np.mean(observation_counts[observation_counts > 0]) if len(observed_landmarks) > 0 else 0
    
    summary_text = f"""Coverage Summary:
    • Observed: {len(observed_landmarks)}/{NUM_LANDMARKS} ({coverage_pct:.1f}%)
    • Unobserved: {len(unobserved_landmarks)}
    • Avg Observations: {avg_obs:.1f}
    • Max Observations: {np.max(observation_counts)}
    • Min Observations: {np.min(observation_counts[observation_counts > 0]) if len(observed_landmarks) > 0 else 0}
    
    Configuration:
    • Trajectory: {TRAJECTORY_TYPE.capitalize()}
    • Scale: {FIGURE8_SCALE}m
    • Sensor Range: {MAX_RANGE}m
    • Duration: {SIM_TIME}s"""
    
    ax2.text(0.98, 0.97, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    os.makedirs('outputs/analysis', exist_ok=True)
    plt.savefig('outputs/analysis/coverage_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: outputs/analysis/coverage_visualization.png")
    
    plt.show()


if __name__ == "__main__":
    print("Running simulation to analyze coverage...")
    landmarks, ekf, observation_counts = run_simulation()
    print(f"✓ Simulation complete")
    print(f"  Observed: {np.sum(observation_counts > 0)}/{NUM_LANDMARKS} landmarks")
    
    print("\nCreating visualization...")
    visualize_coverage(landmarks, ekf, observation_counts)
    print("Done!")
