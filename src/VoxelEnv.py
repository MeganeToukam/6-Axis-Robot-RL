import numpy as np
import gym
from gym import spaces
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from itertools import product
import plotly.graph_objects as go


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv


# Define DH parameters for the 6-axis robot
DH_PARAMS = np.array([
    [0, 0.15185, np.pi/2],      # θ1
    [-0.24355, 0, 0],           # θ2
    [-0.2132, 0, 0],            # θ3
    [0, 0.13105, np.pi/2],      # θ4
    [0, 0.08535, -np.pi/2],     # θ5
    [0, 0.0921, 0]              # θ6
])

# Forward Kinematics Function
def dh_transformation_matrix(a, d, alpha, theta):
    """Computes the transformation matrix using DH parameters."""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,             np.sin(alpha),                 np.cos(alpha),                 d],
        [0,             0,                             0,                             1]
    ])
 

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Define action space: Each joint can [-0.1°, 0°, +0.1°]
        self.action_space = spaces.MultiDiscrete([3] * 6)  # 3 actions for each of the 6 joints

        # Discretized state space (10x10x10 grid)
        self.grid_size = 100 # Number of bins per axis
        dim = 0.3
        self.state_space_bins = np.linspace(-dim, dim, self.grid_size)

        # Define observation space (Voxel) boundaries
        self.observation_space_analog = spaces.Box(
            low=np.array([-dim, -dim, -dim]), # in m 
            high=np.array([dim, dim, dim]), 
            dtype=np.float32
        ) 
        self.observation_space_discrete = spaces.Discrete(self.grid_size ** 3)

        # Initialize joint angles
        self.joint_angles = np.zeros(6)

        #define global orientation
        self.orientation = np.radians([90, 45, -60])

        # Initialize Voxel grid
        self.observation_space = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=int)
        
        #get indexes of the trajectory
        x0 = 0.1
        y0 = 0
        z0 = 0.1
        a = 0.05
        b = 0.03
        points = self.grid_size
        sphere_radius = 0.01
        sphere_points = 10
        self.trajectory, self.tarjectory_center_points = self.generate_donut_ellipse_spheres(x0, y0, z0, a, b, sphere_radius, self.orientation, sphere_points, self.grid_size)
        discrete_spheres = self.discretize_ellipse_coordinates(self.trajectory, self.state_space_bins)

        # Speichere die ersten 10, letzten 10 und den Rest separat
        n = int(sphere_points**2) # Number of points to extract (exactly the one spheres at begin and end)
        self.trajectory_begin = discrete_spheres[:n]  # Erste 20 Punkte
        self.on_trajectory = discrete_spheres[n:-n]  # Letzte 10 Punkte
        self.trajectory_end = discrete_spheres[-n:]  # Alle Punkte dazwischen

        # set all points on the trajectory to 1
        for i in self.on_trajectory:
            self.observation_space[i[0], i[1], i[2]] = 1

        #set all points on the start of the trajectory to 2
        for i in self.trajectory_begin:
            self.observation_space[i[0], i[1], i[2]] = 2

        #set all points on the end of the trajectory to 3
        for i in self.trajectory_end:
            self.observation_space[i[0], i[1], i[2]] = 3


        self.current_step = 0  # Track progress
        self.reward = 0
        self.step_positions = [] # store all step positions taken by the TCP

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        self.old_joint_angles = np.zeros(6)

    def forward_kinematics(self, joint_angles):
        """Computes the TCP (end-effector) position using direct kinematics."""
        T = np.eye(4)
        orientation = np.zeros(3)
        for i in range(6):
            a, d, alphai = DH_PARAMS[i]
            theta = joint_angles[i]
            #T = np.dot(T, dh_transformation_matrix(a, d, alphai, theta))
            T = T @ dh_transformation_matrix(a, d, alphai, theta)
        alpha = np.arctan2(T[3, 2], T[3,3])
        gamma = np.arctan2(T[2, 1], T[1, 1])
        beta = np.arctan2(-T[3, 1], T[1,1]*np.cos(gamma)+T[2, 1]*np.sin(gamma))
        orientation = alpha, beta, gamma
        return T[:3, 3], orientation # Extract (x, y, z, alpha, beta, gamma) position

    def step(self, action):
        """Applies action, computes new state, gives reward, and checks termination."""
        
        # Convert actions to angle updates
        # action is array of the 6 join angles [0, 1, 2] corresponding to [-1, 0, 1]*0.1°
        delta_angles = (np.array(action) - 1) * 10 * (np.pi / 180)  # Convert degrees to radians
        self.joint_angles += delta_angles

        # Apply joint constraints from -180° to 180°
        self.joint_angles = np.clip(self.joint_angles, np.radians(-180), np.radians(180))

        #print("Joint Angles: ", self.joint_angles)
        # Compute new TCP position
        tcp_position, orientation = self.forward_kinematics(self.joint_angles)

        #print("TCP: ", tcp_position, orientation)

        # save position
        self.step_positions.append(tcp_position)

        #convert position to discrete state
        #state = self.get_discrete_state(tcp_position)

        self.current_step += 1

        # Compute reward
        reward, done = self.compute_reward(tcp_position, orientation)

        #Return state, reward, done, and info
        #return tcp_position, reward, done, {}
        self.old_joint_angles = self.joint_angles
        position = np.append(tcp_position, orientation)
        return position, reward, done, {}
    
    def get_discrete_state(self, position):
        """Convert continuous position to discrete state index."""
        idx = [np.digitize(position[i], self.state_space_bins) - 1 for i in range(3)]
        idx = np.clip(idx, 0, self.grid_size - 1)  # Ensure indices are within bounds
        return np.ravel_multi_index(idx, (self.grid_size, self.grid_size, self.grid_size))

    def compute_reward(self, tcp_position, orientation):
        idx = [np.digitize(tcp_position[i], self.state_space_bins) - 1 for i in range(3)]
        #idx = np.clip(idx, 0, self.grid_size - 1)  # Ensure indices are within bounds

        #if index is out ouf bound
        """ if idx[0] < 0 or idx[1] < 0 or idx[2] < 0 or idx[0] >= self.grid_size or idx[1] >= self.grid_size or idx[2] >= self.grid_size:
            self.reward = 0
            #done = True
            #return self.reward, done
            self.reset() """

        done = False

        #if position idx is the end point of the ellipse, reward is 100
        if self.observation_space[idx[0], idx[1], idx[2]] == 3:
            self.reward += 10
            done = True
        
        #if position idx is on ellipse, reward is -1
        elif self.observation_space[idx[0], idx[1], idx[2]] == 1:
            self.reward -=1

        #if position idx is the start point of the ellipse, reward is 10
        elif self.observation_space[idx[0], idx[1], idx[2]] == 2:
            self.reward -= 2
            #self.export_start_point(self.joint_angles)

        #if position idx is not on the ellipse, reward is -10
        elif self.observation_space[idx[0], idx[1], idx[2]] == 0 and self.observation_space_analog.contains(tcp_position):
            self.reward -= 5

        elif not self.observation_space_analog.contains(tcp_position):
            self.reward -= 20  # Large penalty
            self.reset()
            #print("\n\n Reset ------------------------------------------\n\n")
            #done = True        
        return self.reward, done

    def reset(self):
        """Resets the environment for a new episode."""
        # Positioning af the begin of the trajectory
        self.joint_angles = np.zeros(6)
        #self.joint_angles = np.radians([49.69, -118.87, -118.86, -122.54, 49.69, 0.04])
        #choose random starting point on the ellipse
        #self.joint_angles = np.random.randint(-180, 180, 6)

        """rng = np.random.default_rng()
        self.joint_angles = rng.integers(-180, 180, 6)
        self.joint_angles = np.radians(self.joint_angles)
        self.current_step = 0"""

        # read joint angles corresponding to the start point of the ellipse from file
        seed_angles = read_angles_from_file('joint_angles_at_start_point.txt')
        # Generate random joint angles (6) by picking one random line (array)
        self.joint_angles = np.radians(pick_random_line(seed_angles))
        #self.reward = 0
        # determine joint angles based on initial position of robot, the TCP, starting point of the ellipse
        position, orientation = self.forward_kinematics(self.joint_angles)
        #position = self.trajectory[0]
        return np.append(position, orientation)#elf.get_discrete_state(position)
        #return self.trajectory[0]

    def render(self, tcp_pos, once, mode='human'):
        #self.ax.clear()
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 1])
        self.ax.scatter(tcp_pos[0], tcp_pos[1], tcp_pos[2], c='r', marker='o', label='TCP Position')
        trajectory = self.trajectory
        if once:
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Target Trajectory')
            self.ax.legend()
        plt.draw()
        plt.pause(0.01)


    def visualize(self):
        # Create a 3D plot to visualize the robot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        last_1000_positions = self.step_positions[-1000:] if len(self.step_positions) > 1000 else self.step_positions


        # Extract trajectory data
        x = [point[0] for point in last_1000_positions]
        y = [point[1] for point in last_1000_positions]
        z = [point[2] for point in last_1000_positions]

        # Plot trajectory
        #ax.plot(x, y, z, color='b', linewidth=2, marker='*', markersize=2)
        ax.scatter(x, y, z, color='b', marker='*', s=50, label='Step Positions')

        #ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], self.trajectory[:, 2], color='r', linewidth=2, marker='o', markersize=2)

        # Labels and View
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        #ax.view_init(elev=20., azim=30)  # Adjust viewing angle

        plt.show()

        #later plot the actual robotposition, calculated from the joint angles
        # always plot one point for the position


    def generate_ellipse_coordinates(self, x0, y0, z0, a, b, rot_angles, points):

        alpha, beta, gamma = rot_angles # Rotation angles in radians

        # Generate ellipse points in local coordinates
        t = np.linspace(0, 3 * np.pi/2, points)  # Parameter t
        x_local = a * np.cos(t)
        y_local = b * np.sin(t)
        """Ellipse initially in the XY plane """
        z_local = np.zeros_like(t)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])

        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx  # Matrix multiplication in correct order

        # Apply rotation
        rotated_coords = R @ np.array([x_local, y_local, z_local])

        # Translate to center (x0, y0, z0)
        x, y, z = rotated_coords[0, :] + x0, rotated_coords[1, :] + y0, rotated_coords[2, :] + z0

        trajectory = np.array([x, y, z]).T
        return trajectory
        

    def generate_donut_ellipse_spheres(self, x0, y0, z0, a, b, sphere_radius, rot_angles, sphere_points, ellipse_points):
        """
        Generates an elliptical base with spheres placed around each point of the ellipse.
        """
        alpha, beta, gamma = rot_angles  # Rotation angles

        # Generate ellipse points
        t_ellipse = np.linspace(0, 3 * np.pi/2, ellipse_points)
        x_ellipse = a * np.cos(t_ellipse)
        y_ellipse = b * np.sin(t_ellipse)
        z_ellipse = np.zeros_like(t_ellipse)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])

        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx  # Final rotation matrix

        # Apply rotation
        rotated_coords = R @ np.array([x_ellipse, y_ellipse, z_ellipse])

        # Translate to center
        x_ellipse, y_ellipse, z_ellipse = rotated_coords[0, :] + x0, rotated_coords[1, :] + y0, rotated_coords[2, :] + z0

        all_spheres = []  # Store all sphere coordinates

        # Generate spheres around each ellipse point
        for ex, ey, ez in zip(x_ellipse, y_ellipse, z_ellipse):
            u = np.linspace(0, 2 * np.pi, sphere_points)
            v = np.linspace(0, np.pi, sphere_points)
            x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v)) + ex
            y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v)) + ey
            z_sphere = sphere_radius * np.outer(np.ones_like(u), np.cos(v)) + ez
            all_spheres.append((x_sphere, y_sphere, z_sphere))

        ellipse_coordinates = np.column_stack((x_ellipse, y_ellipse, z_ellipse))  # Return list of sphere coordinates

        return all_spheres, ellipse_coordinates  # Return list of sphere coordinates
    

    def discretize_ellipse_coordinates(self, all_spheres, state_space_bins):
        """
        Finds the indices of the donut_spheres points in the discretized state space grid.
        
        Parameters:
            all_spheres: List of (x, y, z) sphere coordinates.
            state_space_bins: 1D array of discrete values to map coordinates onto.
        
        Returns:
            A list of (ix, iy, iz) indices for each point in the state space grid.
        """
        indices_list = []  # Store the indices of each point

        for x_sphere, y_sphere, z_sphere in all_spheres:
            # Flatten the arrays
            x_flat = x_sphere.flatten()
            y_flat = y_sphere.flatten()
            z_flat = z_sphere.flatten()

            # Find the nearest indices in state_space_bins
            ix = np.digitize(x_flat, state_space_bins) - 1
            iy = np.digitize(y_flat, state_space_bins) - 1
            iz = np.digitize(z_flat, state_space_bins) - 1

            # Clip indices to ensure they are within bounds
            ix = np.clip(ix, 0, len(state_space_bins) - 1)
            iy = np.clip(iy, 0, len(state_space_bins) - 1)
            iz = np.clip(iz, 0, len(state_space_bins) - 1)

            # Store the indices
            indices_list.extend(zip(ix, iy, iz))

        return np.array(indices_list)  # Convert to NumPy array
    
    def export_start_point(self, joint_angles):
    # Export the start position (TCP coordinates) to a CSV file
        """Save the joint angles to a file."""
        # Convert joint angles from radians to degrees for better readability
        joint_angles_degrees = np.degrees(joint_angles)
        
        # Save joint angles to a text file (you can also use CSV or other formats)
        with open('joint_angles.txt', 'a') as file:
            file.write(f"Joint Angles (degrees): {joint_angles_degrees.tolist()}\n")

import random 
def read_angles_from_file(filename):
    angles = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove the square brackets and split by commas, then convert each to float
            line = line.strip()[1:-1]  # Remove the leading '[' and trailing ']'
            angles.append([float(angle) for angle in line.split(',')])
    return angles

# Function to randomly pick one entire line (array) from the seed set
def pick_random_line(seed_angles):
    # Randomly select one entire array (line) from the seed angles
    selected_line = random.choice(seed_angles)
    
    return selected_line


            


