import gymnasium as gym
import matplotlib.pyplot as plt

def render(env, ax):
    """
    Renders the current state of the given environment and displays it as an image.

    Parameters:
        env (gym.Env): The environment to render. It should be initialized with `render_mode='rgb_array'`.
        ax (matplotlib.axes.Axes): The Axes object to draw the image on.
    """
    state_image = env.render()
    ax.imshow(state_image)
    ax.axis('off')
    plt.draw()
    plt.pause(0.5)

# Create environment
env = gym.make('FrozenLake-v1', render_mode='rgb_array')

# Reset the environment to get the initial state
state, info = env.reset(seed=42)

# Manually make the agent reach the goal
# Define the sequence of actions: right, right, down, down, down, right
actions = [2, 2, 1, 1, 1, 2]

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

for action in actions:
    # Execute each action
    state, reward, terminated, truncated, info = env.step(action)
    # Render the environment
    render(env, ax)
    if terminated or truncated:
        print("You reached the goal!")
        break

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
env.close()
