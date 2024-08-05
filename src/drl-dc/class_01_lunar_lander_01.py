
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Initiate the Lunar Lander environment
env = gym.make("LunarLander-v2")

class Network(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(Network, self).__init__()
        # Define a linear transformation layer 
        self.linear = nn.Linear(dim_inputs, dim_outputs)
    def forward(self, x):
        return self.linear(x)

# Instantiate the network
network = Network(dim_inputs=8, dim_outputs=4)

# Initialize the optimizer
optimizer = optim.Adam(network.parameters(), lr=0.0001)

print("Network initialized as:\n", network)


# Select action and calculate loss functions:
def select_action(network, state):
    state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        action_probs = network(state)
    action = torch.argmax(action_probs).item()
    return action

def calculate_loss(network, state, action, next_state, reward, done):
    # Implement your loss calculation here
    # For example, you can use Mean Squared Error Loss
    criterion = nn.MSELoss()
    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)
    
    # Example: Simple TD target for Q-learning
    target = reward + (1.0 - done) * torch.max(network(next_state))
    prediction = network(state)[action]
    
    loss = criterion(prediction, target)
    return loss



env = gym.make("LunarLander-v2")
# Run ten episodes
for episode in range(1, 10):
    state, info = env.reset()
    done = False    
    # Run through steps until done
    while not done:
        # choose action (Policy)
        action = select_action(network, state)        
        # Take the action (run the model)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated        
        # compute loss
        loss = calculate_loss(network, state, action, next_state, reward, done)
        # learn (based on loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        # Update the state
        state = next_state
    print(f"Episode {episode} complete.")