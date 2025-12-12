"""
3D SLAM Visualization with actual 3D robot and environment
Uses PyGame and OpenGL to render a real 3D scene
"""
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from config.params import *
from src.robot import Robot, TrajectoryController
from src.ekf_slam import EKF_SLAM
from src.data_association import DataAssociation
from src.utils import generate_random_landmarks

# 3D Rendering Functions
def draw_cube(size=0.5, color=(1, 0, 0)):
    """Draw a cube (for robot body)"""
    glColor3f(*color)
    vertices = [
        [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],
        [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    faces = [
        (0,1,2,3), (3,2,6,7), (7,6,5,4),
        (4,5,1,0), (1,5,6,2), (4,0,3,7)
    ]
    
    # Draw filled faces
    glBegin(GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    # Draw edges
    glColor3f(0, 0, 0)
    glLineWidth(2)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_cylinder(radius=0.2, height=1.0, color=(0, 1, 0)):
    """Draw a cylinder (for landmarks)"""
    glColor3f(*color)
    quadric = gluNewQuadric()
    gluCylinder(quadric, radius, radius, height, 20, 20)

def draw_cone(radius=0.3, height=0.5, color=(1, 0, 0)):
    """Draw a cone (for robot direction indicator)"""
    glColor3f(*color)
    quadric = gluNewQuadric()
    gluCylinder(quadric, radius, 0, height, 20, 20)

def draw_sphere(radius=0.3, color=(0, 0, 1)):
    """Draw a sphere (for estimated landmarks)"""
    glColor3f(*color)
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, 20, 20)

def draw_grid(size=50, step=2):
    """Draw ground grid"""
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, -size, 0)
        glVertex3f(i, size, 0)
        glVertex3f(-size, i, 0)
        glVertex3f(size, i, 0)
    glEnd()

def draw_trajectory(trajectory, color=(0, 1, 0), height=0.1):
    """Draw trajectory as a line"""
    glColor3f(*color)
    glLineWidth(3)
    glBegin(GL_LINE_STRIP)
    for point in trajectory:
        glVertex3f(point[0], point[1], height)
    glEnd()

def draw_robot_3d(x, y, theta, size=1.0):
    """Draw the robot as a tank-like vehicle"""
    glPushMatrix()
    glTranslatef(x, y, 0.5)  # Position at x, y, lifted 0.5 units
    glRotatef(np.degrees(theta), 0, 0, 1)  # Rotate around Z axis
    
    # Robot body (red cube)
    draw_cube(size=size*0.5, color=(0.8, 0.2, 0.2))
    
    # Direction indicator (cone pointing forward)
    glPushMatrix()
    glTranslatef(size*0.6, 0, 0)  # Move to front
    glRotatef(90, 0, 1, 0)  # Point cone forward
    draw_cone(radius=size*0.2, height=size*0.5, color=(1, 0, 0))
    glPopMatrix()
    
    # Wheels (small dark cubes)
    wheel_positions = [
        (size*0.3, size*0.5, -size*0.3),
        (size*0.3, -size*0.5, -size*0.3),
        (-size*0.3, size*0.5, -size*0.3),
        (-size*0.3, -size*0.5, -size*0.3),
    ]
    for wx, wy, wz in wheel_positions:
        glPushMatrix()
        glTranslatef(wx, wy, wz)
        draw_cube(size=size*0.15, color=(0.2, 0.2, 0.2))
        glPopMatrix()
    
    glPopMatrix()

def draw_landmark_3d(x, y, is_true=True):
    """Draw a landmark as a tree-like structure"""
    glPushMatrix()
    glTranslatef(x, y, 0)
    
    if is_true:
        # True landmark: Green tree
        # Trunk (brown cylinder)
        glColor3f(0.4, 0.2, 0.1)
        quadric = gluNewQuadric()
        gluCylinder(quadric, 0.15, 0.15, 1.5, 10, 10)
        
        # Foliage (green sphere on top)
        glTranslatef(0, 0, 1.5)
        draw_sphere(radius=0.5, color=(0.2, 0.8, 0.2))
    else:
        # Estimated landmark: Blue pyramid
        glTranslatef(0, 0, 0.5)
        draw_sphere(radius=0.4, color=(0.3, 0.3, 1.0))
    
    glPopMatrix()

def draw_sensor_cone(x, y, theta, fov_angle, max_range):
    """Draw sensor field of view as a transparent cone"""
    glPushMatrix()
    glTranslatef(x, y, 0.3)
    glRotatef(np.degrees(theta), 0, 0, 1)
    
    # Enable transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1, 1, 0, 0.2)  # Yellow with transparency
    
    # Draw cone shape
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)
    
    num_segments = 20
    for i in range(num_segments + 1):
        angle = -fov_angle/2 + (fov_angle * i / num_segments)
        x_cone = max_range * np.cos(angle)
        y_cone = max_range * np.sin(angle)
        glVertex3f(x_cone, y_cone, 0)
    glEnd()
    
    glDisable(GL_BLEND)
    glPopMatrix()

def run_3d_slam_visualization():
    """Main 3D visualization loop"""
    print("="*60)
    print("3D SLAM Real-Time Visualization")
    print("="*60)
    print("\nControls:")
    print("  • Mouse: Rotate camera")
    print("  • Arrow Keys: Pan camera")
    print("  • +/-: Zoom in/out")
    print("  • SPACE: Pause/Resume")
    print("  • ESC: Exit")
    print("\n" + "="*60)
    
    # Initialize PyGame and OpenGL
    pygame.init()
    display = (1600, 900)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D SLAM Visualization")
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Light setup
    glLight(GL_LIGHT0, GL_POSITION, (10, 10, 20, 1))
    glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    
    # Camera setup
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(0, 0, -50)  # Move camera back
    glRotatef(30, 1, 0, 0)  # Tilt down slightly
    
    # Initialize SLAM components
    np.random.seed(42)
    num_landmarks = NUM_LANDMARKS
    true_landmarks = generate_random_landmarks(num_landmarks, LANDMARK_AREA_SIZE)
    
    robot = Robot(INITIAL_STATE, DT)
    
    # Initialize Q-Learning if needed
    qlearning_controller = None
    if CONTROL_TYPE == "qlearning":
        from src.qlearning_controller import QLearningController
        import os
        qlearning_controller = QLearningController(
            num_bins=QL_NUM_BINS,
            num_actions=QL_NUM_ACTIONS,
            alpha=QL_ALPHA,
            gamma=QL_GAMMA,
            epsilon=QL_EPSILON
        )
        if os.path.exists(QL_MODEL_PATH):
            qlearning_controller.load_model(QL_MODEL_PATH)
            print(f"Loaded Q-Learning model from {QL_MODEL_PATH}")
    
    controller = TrajectoryController(TRAJECTORY_TYPE, {
        'radius': CIRCLE_RADIUS,
        'velocity': LINEAR_VELOCITY,
        'scale': FIGURE8_SCALE
    }, control_type=CONTROL_TYPE, qlearning_controller=qlearning_controller)
    
    ekf_slam = EKF_SLAM(INITIAL_STATE, num_landmarks, INITIAL_STATE_COV)
    data_assoc = DataAssociation(ekf_slam)
    
    # Simulation state
    t = 0
    paused = False
    camera_rotation = [0, 0]
    camera_pan = [0, 0, 0]
    zoom = 0
    
    # Store trajectories
    true_trajectory = []
    est_trajectory = []
    
    clock = pygame.time.Clock()
    
    print("\nStarting simulation...")
    
    running = True
    while running and t < SIM_TIME:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"\n{'PAUSED' if paused else 'RESUMED'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    zoom += 2
                elif event.key == pygame.K_MINUS:
                    zoom -= 2
        
        # Handle mouse rotation
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left click
            mouse_rel = pygame.mouse.get_rel()
            camera_rotation[0] += mouse_rel[1] * 0.2
            camera_rotation[1] += mouse_rel[0] * 0.2
        else:
            pygame.mouse.get_rel()  # Clear relative movement
        
        # Handle keyboard pan
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_pan[0] += 0.3
        if keys[pygame.K_RIGHT]:
            camera_pan[0] -= 0.3
        if keys[pygame.K_UP]:
            camera_pan[1] -= 0.3
        if keys[pygame.K_DOWN]:
            camera_pan[1] += 0.3
        
        # Update simulation (if not paused)
        if not paused:
            robot_state = robot.get_state()
            v, w = controller.get_control(robot_state, t)
            robot.move(v, w, add_noise=True)
            ekf_slam.predict(v, w, DT)
            
            measurements, measured_landmark_ids = data_assoc.simulate_measurements(
                robot.get_state(), true_landmarks, add_noise=True
            )
            
            for landmark_id, measurement in zip(measured_landmark_ids, measurements):
                ekf_slam.update(landmark_id, measurement)
            
            # Store trajectories
            true_trajectory.append(robot.get_state()[0:2].copy())
            est_trajectory.append(ekf_slam.get_robot_state()[0:2].copy())
            
            t += DT
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        
        # Apply camera transformations
        glTranslatef(camera_pan[0], camera_pan[1], zoom)
        glRotatef(camera_rotation[0], 1, 0, 0)
        glRotatef(camera_rotation[1], 0, 0, 1)
        
        # Draw ground grid
        glDisable(GL_LIGHTING)
        draw_grid(size=30, step=2)
        glEnable(GL_LIGHTING)
        
        # Draw true landmarks (green trees)
        for i, landmark in enumerate(true_landmarks):
            draw_landmark_3d(landmark[0], landmark[1], is_true=True)
        
        # Draw estimated landmarks (blue spheres)
        for i in range(num_landmarks):
            if ekf_slam.landmark_initialized[i]:
                est_lm = ekf_slam.get_landmark_state(i)
                draw_landmark_3d(est_lm[0], est_lm[1], is_true=False)
        
        # Draw trajectories
        if len(true_trajectory) > 1:
            glDisable(GL_LIGHTING)
            draw_trajectory(true_trajectory, color=(0, 1, 0), height=0.1)
            draw_trajectory(est_trajectory, color=(0, 0, 1), height=0.15)
            glEnable(GL_LIGHTING)
        
        # Draw sensor FOV
        robot_state = robot.get_state()
        glDisable(GL_LIGHTING)
        draw_sensor_cone(robot_state[0], robot_state[1], robot_state[2], 
                        FOV_ANGLE, MAX_RANGE)
        glEnable(GL_LIGHTING)
        
        # Draw robot
        draw_robot_3d(robot_state[0], robot_state[1], robot_state[2], size=1.2)
        
        glPopMatrix()
        
        # Display info text
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], 0, display[1], -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Simple text rendering (white background with info)
        info_text = f"Time: {t:.1f}s | Landmarks: {np.sum(ekf_slam.landmark_initialized)}/{num_landmarks} | " \
                   f"{'PAUSED' if paused else 'RUNNING'}"
        
        # Note: Pygame text rendering in 3D context is complex, so we skip it for now
        # You'll see the info in the console instead
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_LIGHTING)
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
        
        # Console output
        if int(t / DT) % 50 == 0:
            print(f"Time: {t:.1f}s | Landmarks: {np.sum(ekf_slam.landmark_initialized)}/{num_landmarks}", end='\r')
    
    print(f"\n\nSimulation complete! Final time: {t:.1f}s")
    pygame.quit()

if __name__ == "__main__":
    run_3d_slam_visualization()
