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
    Generate scattered landmark positions strategically placed for challenge
    
    Creates landmarks that are:
    - Widely scattered across the environment
    - Some near the trajectory path, some far
    - Forcing robot to rely on sparse observations
    - Testing long-range and short-range sensing
    
    Args:
        num_landmarks: Number of landmarks to generate
        area_size: Size of square area
    
    Returns:
        Array of landmark positions [x, y]
    """
    landmarks = []
    
    # Divide landmarks into different zones for strategic placement
    zone_size = area_size / 3
    
    # Zone 1: Inner zone (near path) - 30% of landmarks
    inner_count = int(num_landmarks * 0.3)
    for i in range(inner_count):
        r = np.random.uniform(3, 8)  # Distance from center
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        landmarks.append([x, y])
    
    # Zone 2: Mid zone - 40% of landmarks  
    mid_count = int(num_landmarks * 0.4)
    for i in range(mid_count):
        r = np.random.uniform(8, area_size * 0.4)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        landmarks.append([x, y])
    
    # Zone 3: Outer zone (challenging to see) - remaining landmarks
    remaining = num_landmarks - len(landmarks)
    for i in range(remaining):
        # Scattered in outer regions
        x = np.random.uniform(-area_size/2, area_size/2)
        y = np.random.uniform(-area_size/2, area_size/2)
        # Ensure they're in outer zone
        while np.sqrt(x**2 + y**2) < area_size * 0.4:
            x = np.random.uniform(-area_size/2, area_size/2)
            y = np.random.uniform(-area_size/2, area_size/2)
        landmarks.append([x, y])
    
    # Shuffle to avoid any systematic ordering
    landmarks = np.array(landmarks)
    np.random.shuffle(landmarks)
    
    return landmarks


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
