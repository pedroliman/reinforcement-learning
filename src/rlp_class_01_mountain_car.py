import gymnasium as gym
import matplotlib.pyplot as plt
import time

# Create environment
env = gym.make('MountainCar-v0', render_mode='rgb_array')

# Reset the environment to get the initial state
state, info = env.reset(seed=42)

def render(env):
    """
    Renders the current state of the given environment and displays it as an image.

    Parameters:
        env (gym.Env): The environment to render. It should be initialized with `render_mode='rgb_array'`.
    """
    state_image = env.render()
    plt.imshow(state_image)
    plt.axis('off')  # Hide axes for a cleaner display
    plt.show()

# Number of steps to visualize
num_steps = 100

for _ in range(num_steps):
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    state, reward, done, truncated, info = env.step(action)
    
    # Render the environment and visualize
    render(env)
    
    # Adding a small delay to better visualize the movement
    time.sleep(0.1)
    
    # If the episode is done, reset the environment
    if done or truncated:
        state, info = env.reset()

# Close the environment when done
env.close()
