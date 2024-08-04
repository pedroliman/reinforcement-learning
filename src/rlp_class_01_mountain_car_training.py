import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import numpy as np

# Create environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 20
num_steps = 200

# Discretization parameters
state_bins = [20, 20]  # Number of bins for each state dimension
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.07, 0.07]  # Clip velocity bounds for discretization

# Initialize Q-table
q_table = np.random.uniform(low=-1, high=1, size=(state_bins + [env.action_space.n]))

def discretize_state(state):
    ratios = [(state[i] + abs(state_bounds[i][0])) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    new_state = [int(round((state_bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(state_bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def render(env, episode):
    """
    Renders the current state of the given environment, displays the episode number, and returns it as an image.

    Parameters:
        env (gym.Env): The environment to render. It should be initialized with `render_mode='rgb_array'`.
        episode (int): The current episode number.
    
    Returns:
        numpy.ndarray: The rendered image of the current state with the episode number.
    """
    state_image = env.render()
    plt.imshow(state_image)
    plt.axis('off')
    plt.title(f"Episode: {episode}")
    plt.savefig('frame.png')
    plt.close()
    return imageio.imread('frame.png')

# List to store frames for the GIF
frames = []

for episode in range(1, num_episodes + 1):
    state, info = env.reset(seed=42)
    state = discretize_state(state)
    total_reward = 0

    for _ in range(num_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        # Update Q-table
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state

        # Capture the frame
        frames.append(render(env, episode))

        if done or truncated:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Close the environment
env.close()

# Save frames as a GIF
gif_path = 'mountain_car_training.gif'
imageio.mimsave(gif_path, frames, fps=10)

print(f"GIF saved as {gif_path}")
