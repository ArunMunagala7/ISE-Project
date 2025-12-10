# SLAM Project Implementation Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation
```bash
python main.py
```

### 3. Run with Custom Parameters
```bash
# Circle trajectory with 20 landmarks
python main.py --trajectory circle --num_landmarks 20

# Figure-8 trajectory
python main.py --trajectory figure8 --num_landmarks 15

# Save video output
python main.py --save_video

# Custom random seed
python main.py --seed 123
```

## Project Structure Explained

```
ise-project/
├── config/
│   └── params.py              # All simulation parameters (noise, sensor, etc.)
│
├── src/
│   ├── robot.py               # Robot motion model (unicycle dynamics)
│   ├── ekf_slam.py           # Core EKF-SLAM algorithm
│   ├── data_association.py   # Measurement-to-landmark matching
│   ├── visualization.py      # Real-time plotting and animation
│   └── utils.py              # Helper functions
│
├── outputs/
│   ├── plots/                # Generated convergence plots
│   └── videos/               # Animation videos (if enabled)
│
├── main.py                   # Main simulation script
└── requirements.txt          # Python dependencies
```

## Implementation Details

### Core Components

#### 1. Robot Motion Model (`src/robot.py`)
- **Unicycle dynamics**: Standard differential drive model
- **Equations**:
  ```
  x_{k+1} = x_k + v*cos(θ)*dt
  y_{k+1} = y_k + v*sin(θ)*dt  
  θ_{k+1} = θ_k + ω*dt
  ```
- **Noise**: Gaussian noise on velocity commands
- **Controller**: Feedback controller to follow desired trajectory

#### 2. EKF-SLAM (`src/ekf_slam.py`)
Main algorithm with two key steps:

**Prediction Step:**
- Applies motion model to predict robot state
- Computes Jacobian for linearization
- Updates covariance with process noise

**Update Step:**
- Receives range-bearing measurements
- Computes measurement Jacobian
- Uses Kalman gain to correct state estimate
- Updates covariance

**State Vector Structure:**
```
[x, y, θ, lx_1, ly_1, lx_2, ly_2, ..., lx_n, ly_n]
 \_____/  \___________________________________/
  Robot              Landmarks
```

#### 3. Data Association (`src/data_association.py`)
- Determines which measurement belongs to which landmark
- Uses **Mahalanobis distance** with gating
- Handles sensor FOV and range limitations
- Initializes new landmarks on first observation

#### 4. Visualization (`src/visualization.py`)
- Real-time plotting of:
  - True vs estimated robot trajectory
  - True vs estimated landmark positions
  - Uncertainty ellipses (covariance)
  - Convergence error over time

### Key Mathematical Concepts

#### Jacobians
The EKF linearizes the nonlinear motion model using Jacobians:

**Motion Model Jacobian (G):**
```
∂f/∂state = [1   0   -v*sin(θ)*dt]
            [0   1    v*cos(θ)*dt]
            [0   0    1           ]
```

**Measurement Model Jacobian (H):**
For range-bearing sensor:
```
range = √((lx-x)² + (ly-y)²)
bearing = atan2(ly-y, lx-x) - θ
```

Jacobian computed using partial derivatives.

#### Uncertainty Representation
- **Covariance matrix**: Captures uncertainty in state estimate
- **Uncertainty ellipses**: Visual representation (95% confidence)
- Uncertainty decreases as robot observes landmarks multiple times

## Configuring Parameters

Edit `config/params.py` to tune the simulation:

### Motion Noise
```python
MOTION_NOISE_V = 0.1   # Linear velocity noise (m/s)
MOTION_NOISE_W = 0.05  # Angular velocity noise (rad/s)
```

### Measurement Noise
```python
MEASUREMENT_NOISE_RANGE = 0.3    # Range noise (m)
MEASUREMENT_NOISE_BEARING = 0.1  # Bearing noise (rad)
```

### Sensor Parameters
```python
MAX_RANGE = 10.0      # Maximum sensor range (m)
FOV_ANGLE = np.pi     # Field of view (rad)
```

### Trajectory
```python
TRAJECTORY_TYPE = "circle"  # or "figure8"
CIRCLE_RADIUS = 5.0
LINEAR_VELOCITY = 1.0
```

## Expected Results

### Visualization
You should see:
1. **Left plot**: Map view showing:
   - Green stars: True landmark positions
   - Blue triangles: Estimated landmarks
   - Blue ellipses: Landmark uncertainty
   - Green line: True robot trajectory
   - Blue dashed: Estimated trajectory
   - Cyan ellipse: Robot position uncertainty

2. **Right plot**: Error convergence
   - Blue line: Robot position error
   - Red line: Average landmark error
   - Both should decrease over time

### Convergence
- Initially: Large uncertainty, noisy estimates
- Over time: Uncertainty shrinks, estimates converge to truth
- After 1-2 laps: Significant improvement
- After 3+ laps: Near-optimal performance

## Analysis & Report

### What to Include in Technical Report

1. **Introduction**
   - Problem statement (SLAM)
   - Approach (EKF-SLAM)

2. **Mathematical Framework**
   - State representation
   - Motion model equations
   - Measurement model equations
   - EKF prediction derivation
   - EKF update derivation
   - Jacobian calculations

3. **Implementation**
   - Algorithm pseudocode
   - Data association strategy
   - Parameter choices

4. **Results**
   - Convergence plots
   - Final error statistics
   - Discussion of noise effects
   - Comparison of different scenarios

5. **Code Appendix**
   - Key code snippets
   - Full code reference

### Generating Plots for Report

The simulation automatically saves:
- `outputs/plots/convergence_error.png` - Error over time

For additional analysis:
```python
# Modify main.py to save more plots
# Add after simulation completes:
visualizer.save_final_plots('outputs/plots')
```

## Experiments to Try

1. **Vary noise levels**: 
   - Increase/decrease motion and measurement noise
   - Observe effect on convergence rate

2. **Different trajectories**:
   - Circle vs figure-8
   - Which provides better observability?

3. **Sensor limitations**:
   - Reduce FOV or max range
   - See impact on mapping quality

4. **Number of landmarks**:
   - More landmarks = more constraints
   - Better localization accuracy

## Troubleshooting

### Simulation doesn't converge
- Check noise parameters (too high?)
- Ensure landmarks are observable
- Verify data association is working

### Plots not displaying
- Make sure matplotlib backend is configured
- Try `plt.show()` at end

### Import errors
- Ensure all files in correct directories
- Run from project root: `python main.py`

## Next Steps for Enhancement

1. **Loop closure detection**: Recognize revisited areas
2. **Unknown correspondences**: Full data association
3. **3D SLAM**: Extend to 3D environment
4. **Different sensors**: Camera, LIDAR
5. **Online mapping**: Unknown number of landmarks

## References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*
- Durrant-Whyte, H., & Bailey, T. (2006). *Simultaneous localization and mapping*
