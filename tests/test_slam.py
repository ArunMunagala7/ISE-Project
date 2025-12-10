"""
Unit tests for SLAM components
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.utils import normalize_angle, generate_random_landmarks
from config.params import DT, INITIAL_STATE, INITIAL_STATE_COV


def test_robot_motion():
    """Test robot motion model"""
    print("Testing robot motion model...")
    
    robot = Robot([0, 0, 0], DT)
    
    # Move forward
    robot.move(1.0, 0.0, add_noise=False)
    state = robot.get_state()
    
    # Should move in x direction
    assert state[0] > 0, "Robot should move forward"
    assert abs(state[1]) < 0.01, "Robot should not move sideways"
    
    print("✓ Robot motion test passed")


def test_angle_normalization():
    """Test angle normalization"""
    print("Testing angle normalization...")
    
    # Test basic cases
    result = normalize_angle(2*np.pi)
    assert abs(result) < 1e-6, f"2π should normalize to ~0, got {result}"
    
    result = normalize_angle(np.pi)
    assert abs(result - np.pi) < 1e-6, f"π should stay π, got {result}"
    
    result = normalize_angle(3*np.pi)
    # 3π wraps to -π or π (both valid)
    assert abs(abs(result) - np.pi) < 1e-6, f"3π should wrap to ±π, got {result}"
    
    print("✓ Angle normalization test passed")


def test_ekf_initialization():
    """Test EKF initialization"""
    print("Testing EKF initialization...")
    
    num_landmarks = 10
    ekf = EKF_SLAM(INITIAL_STATE, num_landmarks, INITIAL_STATE_COV)
    
    # Check dimensions
    expected_dim = 3 + 2 * num_landmarks
    assert len(ekf.mu) == expected_dim, f"State dimension should be {expected_dim}"
    assert ekf.sigma.shape == (expected_dim, expected_dim), "Covariance should be square"
    
    # Check initial state
    assert np.allclose(ekf.mu[0:3], INITIAL_STATE), "Initial robot state incorrect"
    
    # Check no landmarks initialized
    assert np.sum(ekf.landmark_initialized) == 0, "No landmarks should be initialized"
    
    print("✓ EKF initialization test passed")


def test_ekf_prediction():
    """Test EKF prediction step"""
    print("Testing EKF prediction...")
    
    ekf = EKF_SLAM([0, 0, 0], 5, np.eye(3) * 0.1)
    
    # Store initial state
    initial_state = ekf.mu[0:3].copy()
    
    # Predict with forward motion
    ekf.predict(1.0, 0.0, DT)
    
    # Check robot moved
    assert ekf.mu[0] > initial_state[0], "Robot should move forward"
    
    # Check landmarks didn't move
    for i in range(5):
        idx = 3 + 2*i
        assert ekf.mu[idx] == 0, "Landmarks should not move"
    
    print("✓ EKF prediction test passed")


def test_measurement_model():
    """Test measurement model"""
    print("Testing measurement model...")
    
    ekf = EKF_SLAM([0, 0, 0], 1, np.eye(3) * 0.1)
    
    # Robot at origin, landmark at (5, 0)
    robot_pose = np.array([0, 0, 0])
    landmark_pos = np.array([5, 0])
    
    measurement = ekf.measurement_model(robot_pose, landmark_pos)
    
    # Range should be 5
    assert abs(measurement[0] - 5.0) < 1e-10, "Range should be 5"
    
    # Bearing should be 0 (straight ahead)
    assert abs(measurement[1]) < 1e-10, "Bearing should be 0"
    
    print("✓ Measurement model test passed")


def test_landmark_generation():
    """Test landmark generation"""
    print("Testing landmark generation...")
    
    landmarks = generate_random_landmarks(10, 20)
    
    assert landmarks.shape == (10, 2), "Should generate 10 2D landmarks"
    assert np.all(landmarks >= -10), "Landmarks should be within bounds"
    assert np.all(landmarks <= 10), "Landmarks should be within bounds"
    
    print("✓ Landmark generation test passed")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    test_angle_normalization()
    test_landmark_generation()
    test_robot_motion()
    test_ekf_initialization()
    test_ekf_prediction()
    test_measurement_model()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
