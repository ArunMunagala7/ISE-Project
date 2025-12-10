"""
Diagnostic tool to analyze SLAM simulation results
"""
import numpy as np
import matplotlib.pyplot as plt
from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks

print("="*60)
print("SLAM DIAGNOSTIC ANALYSIS")
print("="*60)

# Set seed for reproducibility
np.random.seed(42)

# Setup
num_landmarks = NUM_LANDMARKS
true_landmarks = generate_random_landmarks(num_landmarks, LANDMARK_AREA_SIZE)
robot = Robot(INITIAL_STATE, DT)
controller = TrajectoryController(TRAJECTORY_TYPE, {
    'radius': CIRCLE_RADIUS,
    'velocity': LINEAR_VELOCITY,
    'scale': FIGURE8_SCALE
})
ekf_slam = EKF_SLAM(INITIAL_STATE, num_landmarks, INITIAL_STATE_COV)
data_assoc = DataAssociation(ekf_slam)

print(f"\nConfiguration:")
print(f"  Trajectory: {TRAJECTORY_TYPE}")
print(f"  Landmarks: {num_landmarks}")
print(f"  Simulation time: {SIM_TIME}s")
print(f"  Sensor FOV: {np.degrees(FOV_ANGLE):.0f}°")
print(f"  Sensor range: {MAX_RANGE}m")

# Run simulation
print(f"\nRunning simulation...")
num_steps = int(SIM_TIME / DT)
trajectory_analysis = []

for step in range(num_steps):
    t = step * DT
    robot_state = robot.get_state()
    v, w = controller.get_control(robot_state, t)
    robot.move(v, w, add_noise=True)
    ekf_slam.predict(v, w, DT)
    
    measurements, measured_landmark_ids = data_assoc.simulate_measurements(
        robot.get_state(), true_landmarks, add_noise=True
    )
    
    for landmark_id, measurement in zip(measured_landmark_ids, measurements):
        ekf_slam.update(landmark_id, measurement)
    
    # Collect data every 50 steps
    if step % 50 == 0:
        robot_est = ekf_slam.get_robot_state()
        pos_error = np.linalg.norm(robot.get_state()[0:2] - robot_est[0:2])
        
        trajectory_analysis.append({
            'time': t,
            'true_pos': robot.get_state()[0:2].copy(),
            'est_pos': robot_est[0:2].copy(),
            'pos_error': pos_error,
            'num_initialized': np.sum(ekf_slam.landmark_initialized),
            'num_visible': len(measurements)
        })

# Final analysis
print(f"\n{'='*60}")
print("RESULTS ANALYSIS")
print(f"{'='*60}")

final = trajectory_analysis[-1]
print(f"\nFinal Statistics:")
print(f"  Landmarks discovered: {final['num_initialized']}/{num_landmarks} ({100*final['num_initialized']/num_landmarks:.0f}%)")
print(f"  Final position error: {final['pos_error']:.3f} m")

# Trajectory shape analysis
true_traj = robot.get_trajectory()
print(f"\nTrajectory Analysis:")
print(f"  Total path length: {np.sum(np.linalg.norm(np.diff(true_traj[:,:2], axis=0), axis=1)):.1f} m")
print(f"  X range: [{true_traj[:,0].min():.1f}, {true_traj[:,0].max():.1f}] m")
print(f"  Y range: [{true_traj[:,1].min():.1f}, {true_traj[:,1].max():.1f}] m")

# Check if it's actually a figure-8
x_crossings = np.sum(np.diff(np.sign(true_traj[:,0])) != 0)
y_crossings = np.sum(np.diff(np.sign(true_traj[:,1])) != 0)
print(f"  X-axis crossings: {x_crossings}")
print(f"  Y-axis crossings: {y_crossings}")

if x_crossings > 2 and y_crossings > 2:
    print(f"  ✓ Figure-8 shape detected!")
else:
    print(f"  ✗ NOT a figure-8 (might be straight line or circle)")

# Landmark distribution
print(f"\nLandmark Distribution:")
distances = np.linalg.norm(true_landmarks, axis=1)
print(f"  Distance from origin: min={distances.min():.1f}m, max={distances.max():.1f}m, avg={distances.mean():.1f}m")
print(f"  Landmarks within 8m: {np.sum(distances < 8)}/{num_landmarks}")
print(f"  Landmarks within 12m: {np.sum(distances < 12)}/{num_landmarks}")

# Observation statistics
observations_over_time = [d['num_visible'] for d in trajectory_analysis]
print(f"\nObservation Statistics:")
print(f"  Avg landmarks visible: {np.mean(observations_over_time):.1f}")
print(f"  Max landmarks visible: {np.max(observations_over_time)}")
print(f"  Min landmarks visible: {np.min(observations_over_time)}")

# Convergence analysis
errors_over_time = [d['pos_error'] for d in trajectory_analysis]
print(f"\nError Convergence:")
print(f"  Initial error: {errors_over_time[0]:.3f} m")
print(f"  Final error: {errors_over_time[-1]:.3f} m")
print(f"  Average error: {np.mean(errors_over_time):.3f} m")
if errors_over_time[-1] < errors_over_time[0]:
    print(f"  ✓ Converged! ({100*(errors_over_time[0]-errors_over_time[-1])/errors_over_time[0]:.0f}% improvement)")
else:
    print(f"  ✗ Did not converge")

# Landmark accuracy
lm_errors = []
for i in range(num_landmarks):
    if ekf_slam.landmark_initialized[i]:
        est = ekf_slam.get_landmark_state(i)
        true = true_landmarks[i]
        error = np.linalg.norm(est - true)
        lm_errors.append(error)

if lm_errors:
    print(f"\nLandmark Accuracy:")
    print(f"  Average error: {np.mean(lm_errors):.3f} m")
    print(f"  Max error: {np.max(lm_errors):.3f} m")
    print(f"  Min error: {np.min(lm_errors):.3f} m")

# Create simple visualization
print(f"\n{'='*60}")
print("Creating diagnostic plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectory
ax = axes[0, 0]
ax.plot(true_traj[:, 0], true_traj[:, 1], 'g-', linewidth=2, label='True Path', alpha=0.7)
ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='green', marker='*', s=200, 
          label='True Landmarks', edgecolors='black', linewidths=1, zorder=5)
est_traj = np.array([mu[0:2] for mu in ekf_slam.mu_history])
ax.plot(est_traj[:, 0], est_traj[:, 1], 'b--', linewidth=2, label='Estimated Path', alpha=0.7)
for i in range(num_landmarks):
    if ekf_slam.landmark_initialized[i]:
        lm = ekf_slam.get_landmark_state(i)
        ax.plot(lm[0], lm[1], 'b^', markersize=10)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Trajectory: {TRAJECTORY_TYPE.upper()}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Plot 2: Error over time
ax = axes[0, 1]
times = [d['time'] for d in trajectory_analysis]
errors = [d['pos_error'] for d in trajectory_analysis]
ax.plot(times, errors, 'b-', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position Error (m)')
ax.set_title('Robot Position Error Over Time')
ax.grid(True, alpha=0.3)

# Plot 3: Landmarks visible over time
ax = axes[1, 0]
visible = [d['num_visible'] for d in trajectory_analysis]
initialized = [d['num_initialized'] for d in trajectory_analysis]
ax.plot(times, visible, 'r-', linewidth=2, label='Visible')
ax.plot(times, initialized, 'b-', linewidth=2, label='Initialized')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Number of Landmarks')
ax.set_title('Landmark Observations')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Trajectory X vs Y separately
ax = axes[1, 1]
ax.plot(times, [traj_data['true_pos'][0] for traj_data in trajectory_analysis], 'g-', linewidth=2, label='True X')
ax.plot(times, [traj_data['true_pos'][1] for traj_data in trajectory_analysis], 'g--', linewidth=2, label='True Y')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.set_title('X and Y Position Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Diagnostic plot saved to: outputs/plots/diagnostic_analysis.png")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}\n")
