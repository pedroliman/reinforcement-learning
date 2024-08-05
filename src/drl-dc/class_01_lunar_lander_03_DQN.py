import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim



def describe_episode(episode, reward, episode_reward, t):
    landed = reward == 100
    crashed = reward == -100
    print(
        f"| Episode {episode+1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} |",
        "Landed   |" if landed else "Crashed  |" if crashed else "Hovering |",
    )


# Q Network:
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


state_size = 8
action_size = 4
# Instantiate the Q Network
q_network = QNetwork(state_size, action_size)
# Specify the optimizer learning rate
optimizer = optim.Adam(q_network.parameters(), 0.0001)

print("Q-Network initialized as:\n", q_network)

# Example state:
state = torch.rand(8)

# Policy (Selection_Action):
def select_action(q_network, state):
    # Calculate the Q-values
    q_values = q_network(state)
    print("Q-values:", [round(x, 2) for x in q_values.tolist()])
    # Obtain the action index with highest Q-value
    action = torch.argmax(q_values).item()
    print(f"Action selected: {action}, with q-value {q_values[action]:.2f}")
    return action



select_action(q_network, state)

# Loss Function:
# DQN Loss (Squared Bellman Error):

state = torch.rand(8)
next_state = torch.rand(8)
action = select_action(q_network, state)
reward = 1
gamma = .99
done = False

def calculate_loss(q_network, state, action, next_state, reward, done):
    q_values = q_network(state)
    print(f'Q-values: {q_values}')
    # Obtain the current state Q-value
    current_state_q_value = q_values[action]
    print(f'Current state Q-value: {current_state_q_value:.2f}')
    # Obtain the next state Q-value
    next_state_q_value = q_network(next_state).max()    
    print(f'Next state Q-value: {next_state_q_value:.2f}')
    # Calculate the target Q-value
    target_q_value = reward + gamma * next_state_q_value * (1-done)
    print(f'Target Q-value: {target_q_value:.2f}')
    # Obtain the loss
    loss = nn.MSELoss()(target_q_value, current_state_q_value)
    print(f'Loss: {loss:.2f}')
    return loss

calculate_loss(q_network, state, action, next_state, reward, done)


# Run training Loop:
for episode in range(10):
    state, info = env.reset()
    done = False
    step = 0
    episode_reward = 0
    while not done:
        step += 1     
        # Select the action
        action = select_action(q_network, state)
        next_state, reward, terminated, truncated, _ = (env.step(action))
        done = terminated or truncated
        # Calculate the loss
        loss = calculate_loss(q_network, state, action, next_state, reward, done)
        optimizer.zero_grad()
        # Perform a gradient descent step
        loss.backward()
        optimizer.step()
        state = next_state
        episode_reward += reward
    describe_episode(episode, reward, episode_reward, step)