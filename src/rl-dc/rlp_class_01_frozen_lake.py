from src.aux_funs import render
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode = 'rgb_array')

state, info = env.reset(seed=42)


# Manually make the guy reach the goal:

# Define the sequence of actions
# left, down, right, up
actions = [2,2,1,1,1,2]

plt.ion()  # Turn on interactive mode

for action in actions:
    # Execute each action
    state, reward, terminated, truncated, info = env.step(action)
    # Render the environment
    render(env)
    if terminated or truncated:
        print("You reached the goal!")
        break

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
env.close()