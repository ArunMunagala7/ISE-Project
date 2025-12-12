"""
Utility functions for SLAM implementation
"""
import numpy as np


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi]
    
    Args:
        angle: Angle in radians
    
    Returns:
        Normalized angle in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def compute_mahalanobis_distance(innovation, innovation_cov):
    """
    Compute Mahalanobis distance for data association
    
    Args:
        innovation: Innovation vector (z - z_pred)
        innovation_cov: Innovation covariance matrix
    
    Returns:
        Mahalanobis distance (scalar)
    """
    inv_cov = np.linalg.inv(innovation_cov)
    distance = innovation.T @ inv_cov @ innovation
    return distance


def wrap_to_pi(angles):
    """
    Vectorized angle normalization
    
    Args:
        angles: Array of angles
    
    Returns:
        Normalized angles
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def create_circle_trajectory(radius, num_points=100):
    """
    Generate points for circular trajectory
    
    Args:
        radius: Circle radius
        num_points: Number of points
    
    Returns:
        Array of [x, y] coordinates
    """
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def create_figure8_trajectory(scale=5.0, num_points=200):
    """
    Generate points for figure-8 trajectory
    
    Args:
        scale: Size scale factor
        num_points: Number of points
    
    Returns:
        Array of [x, y] coordinates
    """
    t = np.linspace(0, 2*np.pi, num_points)
    x = scale * np.sin(t)
    y = scale * np.sin(t) * np.cos(t)
    return np.column_stack([x, y])


def generate_random_landmarks(num_landmarks, area_size):
    """
    Generate strategically distributed landmarks for good SLAM coverage
    
    Creates landmarks that are:
    - Some near the trajectory path (easy to observe)
    - Some at medium distance (moderate challenge)
    - Some far away (testing sensor limits)
    
    Args:
        num_landmarks: Number of landmarks to generate
        area_size: Size of square area
    
    Returns:
        Array of landmark positions [x, y]
    """
    landmarks = []
    
    # Zone 1: Inner zone (3-6m from origin) - 40% of landmarks
    inner_count = int(num_landmarks * 0.4)
    for i in range(inner_count):
        r = np.random.uniform(3, 6)  # Close to figure-8 path
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        landmarks.append([x, y])
    
    # Zone 2: Mid zone (6-10m from origin) - 40% of landmarks  
    mid_count = int(num_landmarks * 0.4)
    for i in range(mid_count):
        r = np.random.uniform(6, 10)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        landmarks.append([x, y])
    
    # Zone 3: Outer zone (far landmarks) - remaining 20%
    remaining = num_landmarks - len(landmarks)
    for i in range(remaining):
        x = np.random.uniform(-area_size/2, area_size/2)
        y = np.random.uniform(-area_size/2, area_size/2)
        # Ensure they're in outer zone (> 10m)
        while np.sqrt(x**2 + y**2) < 10:
            x = np.random.uniform(-area_size/2, area_size/2)
            y = np.random.uniform(-area_size/2, area_size/2)
        landmarks.append([x, y])
    
    return np.array(landmarks)


def compute_mse(estimated, true_values):
    """
    Compute Mean Squared Error
    
    Args:
        estimated: Estimated values
        true_values: True values
    
    Returns:
        MSE value
    """
    return np.mean((estimated - true_values) ** 2)
