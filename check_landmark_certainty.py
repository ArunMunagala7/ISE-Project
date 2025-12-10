"""
Check which landmarks have been accurately identified
"""
import numpy as np
from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks

print("="*70)
print("LANDMARK IDENTIFICATION CERTAINTY ANALYSIS")
print("="*70)

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

# Run simulation
print("\nRunning simulation...")
num_steps = int(SIM_TIME / DT)
observation_counts = np.zeros(num_landmarks)

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
        observation_counts[landmark_id] += 1
    
    if step % 100 == 0:
        print(f"  Progress: {100*step/num_steps:.0f}%", end='\r')

print(f"  Progress: 100% - Complete!       ")

# Analyze each landmark
print(f"\n{'='*70}")
print("LANDMARK CERTAINTY REPORT")
print(f"{'='*70}\n")
print(f"{'ID':<4} {'Obs':<6} {'Error(m)':<10} {'Std X':<10} {'Std Y':<10} {'Status':<15}")
print("-"*70)

identified_landmarks = []
uncertain_landmarks = []
never_seen = []

for i in range(num_landmarks):
    if not ekf_slam.landmark_initialized[i]:
        never_seen.append(i)
        print(f"{i:<4} {'0':<6} {'-':<10} {'-':<10} {'-':<10} {'NEVER SEEN':<15}")
        continue
    
    # Get estimate and uncertainty
    est = ekf_slam.get_landmark_state(i)
    true = true_landmarks[i]
    error = np.linalg.norm(est - true)
    
    # Get covariance (uncertainty) for this landmark
    # Landmark i is at indices [3+2*i, 3+2*i+1] in the state vector
    idx_x = 3 + 2*i
    idx_y = 3 + 2*i + 1
    std_x = np.sqrt(ekf_slam.sigma[idx_x, idx_x])
    std_y = np.sqrt(ekf_slam.sigma[idx_y, idx_y])
    avg_std = (std_x + std_y) / 2
    
    # Classification based on uncertainty and observations
    num_obs = int(observation_counts[i])
    
    if avg_std < 0.5 and error < 1.0:
        status = "✓ HIGH CONF"
        identified_landmarks.append((i, num_obs, error, avg_std))
    elif avg_std < 1.0 and error < 2.0:
        status = "~ MEDIUM CONF"
        identified_landmarks.append((i, num_obs, error, avg_std))
    elif avg_std < 2.0:
        status = "? LOW CONF"
        uncertain_landmarks.append((i, num_obs, error, avg_std))
    else:
        status = "✗ VERY UNCERTAIN"
        uncertain_landmarks.append((i, num_obs, error, avg_std))
    
    print(f"{i:<4} {num_obs:<6} {error:<10.3f} {std_x:<10.3f} {std_y:<10.3f} {status:<15}")

# Summary
print("-"*70)
print(f"\nSUMMARY:")
print(f"  High/Medium Confidence: {len(identified_landmarks)}/{num_landmarks} landmarks")
print(f"  Low Confidence: {len(uncertain_landmarks)}/{num_landmarks} landmarks")
print(f"  Never Observed: {len(never_seen)}/{num_landmarks} landmarks")

print(f"\n{'='*70}")
print("WHAT THESE METRICS MEAN:")
print(f"{'='*70}")
print("""
1. Obs (Observations): Number of times the landmark was seen
   - More observations = better estimate
   - Minimum 3-5 needed for good accuracy

2. Error(m): Actual error vs true position (you wouldn't know this in real SLAM!)
   - < 0.5m: Excellent
   - 0.5-1.0m: Good
   - 1.0-2.0m: Fair
   - > 2.0m: Poor

3. Std X/Y (Standard Deviation): Uncertainty in X and Y
   - < 0.5m: High confidence - landmark position well-known
   - 0.5-1.0m: Medium confidence - decent estimate
   - 1.0-2.0m: Low confidence - rough estimate
   - > 2.0m: Very uncertain - barely initialized

4. Status: Overall classification
   ✓ HIGH CONF: Std < 0.5m AND Error < 1.0m
   ~ MEDIUM CONF: Std < 1.0m AND Error < 2.0m
   ? LOW CONF: Std < 2.0m
   ✗ VERY UNCERTAIN: Std > 2.0m

KEY INSIGHT:
In real SLAM, you only know the STD (uncertainty), NOT the actual error!
The STD tells you how confident the system is about that landmark.
""")

if identified_landmarks:
    print(f"\n{'='*70}")
    print("MOST ACCURATELY IDENTIFIED LANDMARKS:")
    print(f"{'='*70}")
    # Sort by average std (lower is better)
    identified_landmarks.sort(key=lambda x: x[3])
    for i, num_obs, error, avg_std in identified_landmarks[:5]:
        print(f"  Landmark {i:2d}: {num_obs:3d} observations, std={avg_std:.3f}m, error={error:.3f}m")

print(f"\n{'='*70}\n")
