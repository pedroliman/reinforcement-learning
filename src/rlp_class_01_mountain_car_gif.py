import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import numpy as np

# Create environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')

# Reset the environment to get the initial state
state, info = env.reset(seed=42)

def render(env):
    """
    Renders the current state of the given environment and returns it as an image.

    Parameters:
        env (gym.Env): The environment to render. It should be initialized with `render_mode='rgb_array'`.
    
    Returns:
        numpy.ndarray: The rendered image of the current state.
    """
    return env.render()

# Number of steps to visualize
num_steps = 100

# List to store frames for the GIF
frames = []

for _ in range(num_steps):
    # Biased random action: more likely to go to the right
    action = np.random.choice([0, 2], p=[0.2, 0.8])
    
    # Step the environment
    state, reward, done, truncated, info = env.step(action)
    
    # Render the environment and capture the frame
    frame = render(env)
    frames.append(frame)
    
    # If the episode is done, reset the environment
    if done or truncated:
        state, info = env.reset()

# Close the environment when done
env.close()

# Save frames as a GIF
gif_path = 'mountain_car.gif'
imageio.mimsave(gif_path, frames, fps=10)

print(f"GIF saved as {gif_path}")
