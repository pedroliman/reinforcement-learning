
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# QNetwork class for reinf. learning.
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Instantiate the first hidden layer
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Instantiate the output layer
        self.fc3 = nn.Linear(64, action_size)
    def forward(self, state):
        # Ensure the ReLU activation function is used
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Setting up the state size and action size
state_size = 8
action_size = 4
# Instantiate the Q Network
q_network = QNetwork(state_size, action_size)
# Specify the optimizer learning rate
optimizer = optim.Adam(q_network.parameters(), 0.0001)

print("Q-Network initialized as:\n", q_network)



def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )
