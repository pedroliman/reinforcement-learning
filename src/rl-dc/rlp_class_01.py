
from src.aux_funs import render
import gymnasium as gym



# CartPole example:

# Create environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')

state, info = env.reset(seed=42)

print(state)

## Taking an action and visualizing the state and reward:

action = 1

state, reward, terminated, _, _ = env.step(action)

reward

# Example: interaction loop for the CartPole example:
while not terminated:
    action = 1 # to the right
    state, reward, terminated, _, _ = env.step(action)
    render()



#nMountain Car Environment:
    
env = gym.make('MountainCar-v0', render_mode = 'rgb_array')

state, info = env.reset(seed=42)

render(env)

