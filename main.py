import numpy as np
from src.VoxelEnv import RobotEnv
import gym
from gym import spaces

import matplotlib.pyplot as plt

from random import random

def get_continuous_position(env, discrete_index):
    """Convert a raveled discrete state index back to a continuous 3D position."""
    
    # Step 1: Unravel the 1D discrete index to 3D grid indices
    idx = np.unravel_index(discrete_index, (env.grid_size, env.grid_size, env.grid_size))

    # Step 2: Compute bin edges for each dimension
    bin_edges = np.linspace(-0.3, 0.3, env.grid_size + 1)

    # Step 3: Compute bin centers (midpoints between edges) for each dimension
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  

    # Step 4: Map discrete indices to their corresponding continuous values
    position = np.array([bin_centers[i] for i in idx])  # Extract corresponding bin centers

    return position

def get_discrete_state(env, position):
        """Convert continuous position to discrete state index."""
        idx = [np.digitize(position[i], env.state_space_bins) - 1 for i in range(3)]
        idx = np.clip(idx, 0, env.grid_size - 1)  # Ensure indices are within bounds
        return np.ravel_multi_index(idx, (env.grid_size, env.grid_size, env.grid_size))




def select_action(env, Q, state, epsilon):
    if np.random.random() < epsilon:
        action_idx = np.random.randint(action_size)
    else:
        action_idx = np.argmax(Q[state])

    # Convert action index to joint movements
    action = np.unravel_index(action_idx, env.action_space.nvec)
    return action





def n_step_Q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, n=5, episodes=1000):
    state_size = env.observation_space_discrete.n
    action_size = np.prod(env.action_space.nvec)

    # Initialize Q-table
    Q = np.zeros((state_size, action_size))

    steps_per_episode = np.zeros(episodes, dtype=int)
    
    for episode in range(episodes):
        position = env.reset()
        state = get_discrete_state(env, position[:3])
        T = float('inf')  # Large number representing max episode length
        states, actions, rewards = [state], [], [0]  # Track states, actions, rewards

        # Select first action using ε-greedy policy
        action = select_action(env, Q, state, epsilon)

        actions.append(action)
        t = 0

        while True:
            if t < T:
                next_position, reward, done, _ = env.step(action)  # Execute action

                next_state = get_discrete_state(env, next_position[:3])
                #print("next_state", next_state)
                #print("next_position", next_position)
                
                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1  # Mark the end of episode
                else:
                    # Select next action using ε-greedy
                    #next_action = select_action(env, Q, next_state, epsilon)
                    next_action = np.argmax(Q[next_state])
                    actions.append(next_action)

            tau = t - n + 1  # Time step for updating

            if tau >= 0:
                # Compute n-step return G
                G = sum([gamma ** (i - tau) * rewards[i] for i in range(tau + 1, min(tau + n, T))])

                if tau + n < T and (tau + n) < len(states) and (tau + n) < len(actions):
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]

                # Update Q-value
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            
            steps_per_episode[episode] += 1

            if tau == T - 1:
                break  # Episode ends

            t += 1
            if t < T:
                action = actions[t]  # Move to next action
            
        """if episode % 100 == 0 or episode % 999 == 0:
            print(f"Episode {episode}, Reward: {reward}")"""

        print(f"Episode {episode}, Reward: {reward}, Steps: {steps_per_episode[episode]}, Position: {next_position}")

    
    return Q, steps_per_episode # Return the learned Q-table


def MonteCArlo_Q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, n=5, episodes=1000):
    state_size = env.observation_space_discrete.n
    action_size = np.prod(env.action_space.nvec)
    Q_table = np.zeros((state_size, action_size))

    steps_per_episode = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        position = env.reset()
        state = get_discrete_state(env, position[:3])
        done = False
        i = 0
        while not done:
            i+=1
            #print("state", state)
            # Select action using ε-greedy strategy
            if np.random.random() < epsilon:
                action_idx = np.random.randint(action_size)
            else:
                action_idx = np.argmax(Q_table[state])

            # Convert action index to joint movements
            action = np.unravel_index(action_idx, env.action_space.nvec)
            #print("action", action)

            # Step in environment
            next_position, reward, done, _ = env.step(action)
            next_state = get_discrete_state(env, next_position[:3])
            #print("Reward: ", reward)

            # Q-learning update
            best_next_action = np.argmax(Q_table[next_state])
            Q_table[state, action_idx] += alpha * (reward + gamma * Q_table[next_state, best_next_action] - Q_table[state, action_idx])

            state = next_state

            
            steps_per_episode[episode] += 1

            # Reduce ε over time (exploration decay)
            #epsilon = max(min_epsilon, epsilon * epsilon_decay)

            #print("state: reward: done: ", state, reward, done)

            # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}")

    return Q_table, steps_per_episode


def evaluate_policy(env, Q, ideal_trajectory, episodes=1000):
    """Evaluate the learned policy using Mean Squared Error (MSE) computed with NumPy.
    
    Args:
        env: The environment.
        Q: The learned Q-table.
        ideal_trajectory: NumPy array of ideal states (trajectory).
        episodes: Number of evaluation episodes.
    
    Returns:
        avg_mse: The average Mean Squared Error over episodes.
    """
    epsilon = 0  # No exploration, follow the learned policy
    mse_list = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset() 
        #trajectory = [state]  # Store the trajectory
        #position = get_continuous_position(env, state)
        trajectory = [position[:3]]  # Store the trajectory
        done = False

        while not done:
            action = np.argmax(Q[state])  # Always exploit the best action
            next_state, _, done, _ = env.step(action)
            position = get_continuous_position(env, next_state)
            trajectory.append(position)
            state = next_state

        # Convert to NumPy arrays
        trajectory = np.array(trajectory)
        ideal = np.array(ideal_trajectory)

        # Ensure both arrays are the same length
        min_length = min(len(trajectory), len(ideal))
        trajectory = trajectory[:min_length]
        ideal = ideal[:min_length]

        # Compute MSE using NumPy
        mse = np.mean((trajectory - ideal) ** 2)
        mse_list[episode] = mse

    avg_mse = np.mean(mse_list)

    return avg_mse, mse_list


env = RobotEnv()

# environment als text speichern
flattened_values = env.observation_space.flatten()

# Generate the 1D indices for each position in the 3D array
indices_1d = np.arange(flattened_values.size)

# Now, let's combine the 1D index with the values (you can reshape for saving purposes)
data_to_save = np.column_stack((indices_1d, flattened_values))

# Save to a text file
np.savetxt("observation_space.txt", data_to_save, fmt="%d", header="Index \t Value")
#np.savetxt("observation_space.txt", env.observation_space.reshape(-1), fmt="%d")

initial_position = env.reset()
print(f"Initial Position: {initial_position}, Discrete State: {get_discrete_state(env, initial_position[:3])}")

# Q-learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize environment
#env = DiscreteRobotEnv()
state_size = env.observation_space_discrete.n
action_size = np.prod(env.action_space.nvec)

# Initialize Q-table
episodes = 100
Q_table = np.zeros((state_size, action_size))
steps_per_episode = np.zeros(episodes)
n = 5
Q_table, steps_per_episode = n_step_Q_learning(env, alpha, gamma, epsilon, 5, episodes)

print("policy training done")
# Plotting the episodes
fig = plt.figure()
plt.plot(list(range(episodes)), steps_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()


""" Evaluate the learned policy """
list_of_mse = np.zeros(episodes)
avg_mse, list_of_mse = evaluate_policy(env, Q_table, env.tarjectory_center_points, episodes)

print(f"Average Mean square error over {episodes} test episodes: {avg_mse:.4f}")

"""Early training benefits from higher alpha, but later, a smaller alpha ensures stability."""
#alpha = max(0.1 / (1 + episode / 500), 0.01)  # Decay over time


# Save trained Q-table


# Plotting the MSE
fig = plt.figure()
plt.plot(list(range(episodes)), list_of_mse)
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()

env.visualize()
