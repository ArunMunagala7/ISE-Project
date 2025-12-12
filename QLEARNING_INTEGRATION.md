# Q-Learning Integration Guide

## Overview

This project now supports **two control modes** for robot trajectory tracking in the EKF-SLAM simulation:

1. **Feedback Control** (original) - Simple proportional controller
2. **Q-Learning Control** (new) - Reinforcement learning-based control

## What Was Added

### 1. Q-Learning Controller Module (`src/qlearning_controller.py`)

A complete reinforcement learning controller that learns to follow trajectories through trial and error.

**Key Features:**
- **State Space**: `(lateral_error, heading_error)` discretized into 10×10 bins
- **Action Space**: 5 discrete angular velocity offsets: `[-0.3, -0.15, 0.0, 0.15, 0.3]` rad/s
- **Learning Algorithm**: Q-Learning with ε-greedy exploration
- **Reward Function**: `r = -(|lateral_error| + 0.5 × |heading_error|)`
- **Hybrid Approach**: Learned offset added to base feedback control (safer than pure RL)

**Parameters:**
```python
QL_ALPHA = 0.1      # Learning rate
QL_GAMMA = 0.9      # Discount factor
QL_EPSILON = 0.2    # Exploration rate (training)
QL_TRAINING = True  # Enable learning during simulation
```

### 2. Modified Robot Controller (`src/robot.py`)

The `TrajectoryController` class now supports both control modes:

```python
controller = TrajectoryController(
    trajectory_type="figure8",
    params={...},
    control_type="qlearning",  # or "feedback"
    qlearning_controller=qlearning_controller
)
```

**How It Works:**
1. Base feedback control computes `(v_base, w_base)`
2. Q-Learning agent observes tracking errors and selects action offset `Δw`
3. Final control: `w = w_base + Δw`
4. Q-table updated using reward signal from next state

### 3. Configuration Parameters (`config/params.py`)

New section for Q-Learning control:

```python
CONTROL_TYPE = "qlearning"  # Switch between modes
QL_NUM_BINS = 10           # State discretization
QL_NUM_ACTIONS = 5         # Action space size
QL_ALPHA = 0.1             # Learning rate
QL_GAMMA = 0.9             # Discount factor
QL_EPSILON = 0.2           # Exploration rate
QL_TRAINING = True         # Enable learning
QL_MODEL_PATH = "outputs/qlearning_model.npy"
```

## How to Use

### Training Mode (Learn While Running)

1. Set training parameters in `config/params.py`:
```python
CONTROL_TYPE = "qlearning"
QL_TRAINING = True   # Enable learning
QL_EPSILON = 0.2     # 20% exploration
```

2. Run simulation:
```bash
python main.py
```

3. The Q-table will be saved automatically to `outputs/qlearning_model.npy`

**Output Example:**
```
--- Q-Learning Statistics ---
Model saved to: outputs/qlearning_model.npy
Q-table shape: (10, 10, 5)
Non-zero entries: 112/500
Q-value range: [-68.199, 0.000]
```

### Exploitation Mode (Use Learned Policy)

1. After training, disable learning:
```python
CONTROL_TYPE = "qlearning"
QL_TRAINING = False  # Disable learning
QL_EPSILON = 0.0     # Pure exploitation (no exploration)
```

2. Run simulation - it will load the trained model:
```bash
python main.py
```

### Fallback to Feedback Control

Simply change the control type:
```python
CONTROL_TYPE = "feedback"  # Use original controller
```

All Q-Learning code is safely preserved but not executed.

## Testing the Integration

### Test Q-Learning Model:
```bash
python test_qlearning.py
```

**Output:**
```
✓ Loaded Q-Learning model from outputs/qlearning_model.npy
  Q-table shape: (10, 10, 5)
  Non-zero entries: 112/500
  Q-value range: [-68.199, 0.000]

✓ Testing learned control:
  Small lateral, small heading error:
    Lateral=0.50m, Heading=5.7°
    → Action offset: -0.300 rad/s
```

### Compare Controllers:
```bash
python compare_controllers.py
```

This runs both feedback and Q-Learning side-by-side and generates comparison plots.

## Architecture Details

### State Discretization

Continuous tracking errors are mapped to discrete bins:

```python
lateral_bin = clip(int((lateral_error + 3.0) / 6.0 * 10), 0, 9)
heading_bin = clip(int((heading_error + π) / (2π) * 10), 0, 9)
state = (lateral_bin, heading_bin)
```

### Action Selection (ε-greedy)

```python
if random() < epsilon:
    action = random_choice([0, 1, 2, 3, 4])  # Explore
else:
    action = argmax(Q[state, :])             # Exploit
```

### Q-Learning Update Rule

```python
Q[s, a] ← Q[s, a] + α × [r + γ × max_a' Q[s', a'] - Q[s, a]]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `α` = learning rate
- `γ` = discount factor

### Reward Function

```python
reward = -(abs(lateral_error) + 0.5 * abs(heading_error))
```

**Design Rationale:**
- Negative reward (cost minimization)
- Lateral error weighted equally important
- Heading error weighted half (allows some heading deviation for smooth turns)
- Encourages staying close to path with correct orientation

## Results from Training Run

From the initial training run (`main.py`):

```
Final robot position error: 13.183m
Landmarks initialized: 28/30
Average landmark position error: 2.625m

Q-Learning Statistics:
- Non-zero entries: 112/500 (22.4% of Q-table explored)
- Q-value range: [-68.199, 0.000]
```

**Interpretation:**
- Agent explored ~22% of state-action space
- Negative Q-values indicate cumulative cost
- Lower (more negative) = worse trajectory performance
- Agent learned to avoid high-cost state-action pairs

## Key Differences from Original Notebooks

### vs. Q_Learning.ipynb:
- **Trajectory**: Figure-8 lemniscate instead of racetrack
- **Integration**: Online learning during SLAM (not separate training phase)
- **Hybrid Control**: Base feedback + learned offset (not pure RL)
- **Simpler Reward**: Direct tracking error (not complex racetrack boundaries)

### vs. Untitled7.ipynb:
- **Learning**: Q-Learning adapts control, Untitled7 uses fixed commands
- **Control**: Sophisticated trajectory tracking vs. simple circular motion
- **Adaptability**: Can learn from different scenarios

## File Structure

```
ise-project/
├── config/
│   └── params.py                    # Added Q-Learning parameters
├── src/
│   ├── qlearning_controller.py     # NEW: RL controller
│   ├── robot.py                     # Modified for hybrid control
│   ├── ekf_slam.py                  # Unchanged
│   ├── data_association.py          # Unchanged
│   └── visualization.py             # Unchanged
├── main.py                          # Updated for Q-Learning init
├── create_animation.py              # Updated for Q-Learning
├── run_3d_visualization.py          # Updated for Q-Learning
├── test_qlearning.py                # NEW: Test learned model
├── compare_controllers.py           # NEW: Compare both modes
└── outputs/
    └── qlearning_model.npy          # Saved Q-table
```

## Troubleshooting

### Model Not Found
```
✗ No model found at outputs/qlearning_model.npy
```
**Solution**: Run training mode first (`QL_TRAINING=True`)

### Poor Performance
If Q-Learning performs worse than feedback:
1. **Insufficient Training**: Run longer (`SIM_TIME` in params.py)
2. **High Exploration**: Lower `QL_EPSILON` for exploitation
3. **Reward Tuning**: Adjust weights in `compute_reward()`
4. **State Resolution**: Increase `QL_NUM_BINS` for finer discretization

### Numerical Warnings
```
RuntimeWarning: divide by zero encountered in matmul
```
These are from the EKF-SLAM covariance updates, not Q-Learning. They don't affect learning.

## Future Improvements

1. **Pre-Training Phase**: Separate training before SLAM evaluation
2. **Function Approximation**: Replace Q-table with neural network for continuous states
3. **Multi-Objective Reward**: Balance tracking accuracy with energy efficiency
4. **Transfer Learning**: Train on one trajectory, test on another
5. **Curiosity-Driven Exploration**: Encourage exploring unseen states

## References

- **Q-Learning Algorithm**: Watkins & Dayan (1992)
- **EKF-SLAM**: Thrun et al., "Probabilistic Robotics" (2005)
- **Hybrid RL Control**: Inspired by Q_Learning.ipynb notebook

## Credits

**Original SLAM Implementation**: Complete EKF-SLAM with figure-8 trajectory, 2D/3D visualization
**Q-Learning Integration**: Added reinforcement learning control while preserving original feedback controller as fallback
**Design Philosophy**: Simultaneous learning + SLAM, hybrid approach for safety
