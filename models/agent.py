import numpy as np              # For numerical operations
import gym                      # For creating the custom gym environment
from gym import spaces          # For defining action and observation spaces
import torch                    # For creating the neural network
import torch.nn as nn           # For neural network layers and functions
import torch.optim as optim     # For optimization algorithms
from torchvision import transforms # For image transformations
from PIL import Image           # For image processing
import os
from collections import deque

# from d4rl_pybullet import CQL  # For using CQL algorithm in d4rl library
# from d4rl_pybullet.envs import RobotEnv 


# Code Structure: 

# 1.Load the .txt file containing the robot's xyz positions
# 2.Load the image file containing the robot's environment
# 3.Define the environment by creating a custom class that inherits from gym.Env
# 4.Implement required methods: __init__, step, reset, and render
# 5.Define the state as the current xyz position, desired position, and image
# 6.Define the action space as discrete with 6 possible actions (±x, ±y, ±z)
# 7.Define the reward as the negative Euclidean distance between the current position and desired position
# 8.Implement a PyTorch reinforcement learning agent
# 9.Train the agent using an appropriate algorithm



# Define the custom Robot environment class which inherits from gym.Env
class RobotEnv(gym.Env):
    # Initialize the environment with necessary parameters
    def __init__(self, positions_file, image_file, desired_position):
        super(RobotEnv, self).__init__()
        self.positions = np.loadtxt(positions_file) # Load robot positions from the file
        self.images = [Image.open(image_file).convert('L') for image_file in image_files] # Load and convert all images to grayscale        
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]) # Define image transformation pipeline
        self.desired_position = np.array(desired_position) # Convert the desired position to a NumPy array
        self.current_position_idx = 0 # Set the current position index to 0
        self.action_space = spaces.Discrete(6) # Define the action space (6 possible actions)
        self.observation_space = spaces.Dict({  # Define the observation space as a dictionary containing two keys
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(6,)), # Define the position space as a Box with 6 dimensions
            'image': spaces.Box(low=0, high=1, shape=(1, 64, 64)) # Define the image space as a Box with dimensions 1x64x64
        })

    # Define the step function, which takes an action and returns the next state, reward, whether the episode is done, and any additional information
    def step(self, action):
        move = np.array([0, 0, 0]) # Initialize a move array with zeros
        move[action // 2] = 1 if action % 2 == 0 else -1 # Determine the move based on the action
        self.positions[self.current_position_idx] += move # Update the current position using the move array
        state = { # Construct the state dictionary
            'position': np.concatenate((self.positions[self.current_position_idx], self.desired_position)), # Concatenate current and desired positions
            'image': self.transform(self.images[self.current_position_idx]) # Apply the image transformation pipeline to the current image
        }
        reward = -np.linalg.norm(self.positions[self.current_position_idx] - self.desired_position) # Calculate the reward as the negative L2 norm of the position difference
        done = self.current_position_idx == len(self.positions) - 1 # Check if the episode is done by comparing the current position index with the total number of positions
        self.current_position_idx += 1 # Increment the current position index
        return state, reward, done, {} # Return the next state, reward, done flag, and an empty info dictionary

    # Define the reset function, which resets the environment to the initial state
    def reset(self):
        self.current_position_idx = 0 # Reset the current position index to 0
        return { # Return the initial state dictionary
            'position': np.concatenate((self.positions[self.current_position_idx], self.desired_position)), # Concatenate current and desired positions
            'image': self.transform(self.images[self.current_position_idx]) # Apply the image transformation pipeline to the first image
        }

    # Define the render function, which is not implemented in this environment


    def render(self, mode='human'):
        pass

from networks import DDQN
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random

class CQLAgent():
    def __init__(self, state_size, action_size, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = 0.99
        
        self.network = DDQN(state_size=(self.state_size,),
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = DDQN(state_size=(self.state_size,),
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-3)
        
    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action
        
    def learn(self, experiences):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

        bellmann_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellmann_error
        
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellmann_error.detach().item()
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


# Helper Functions

def load_image_files(folder_path, file_ext=".png"):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(file_ext)]
    image_files.sort()
    return image_files

def train_cql_agent(agent, env, num_episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=100000):
    replay_buffer = deque(maxlen=buffer_size)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state['position'], epsilon)
            next_state, reward, done, _ = env.step(action[0])

            transition = (torch.tensor(state['position'], dtype=torch.float32).unsqueeze(0),
                          torch.tensor([action], dtype=torch.int64).unsqueeze(1),
                          torch.tensor([reward], dtype=torch.float32).unsqueeze(1),
                          torch.tensor(next_state['position'], dtype=torch.float32).unsqueeze(0),
                          torch.tensor([done], dtype=torch.float32).unsqueeze(1))

            replay_buffer.append(transition)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                experiences = random.sample(replay_buffer, batch_size)
                agent.learn(experiences)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward}")


print ("Loading image files")
image_folder = "/media/quanting/Samsung/Research/Camelmera/models/image_lcam_front"  
image_files = load_image_files(image_folder, file_ext=".png")

print ("Loading trajectory information")
filename = "/media/quanting/Samsung/Research/Camelmera/models/Position_000.txt"
desired_position = [9.939322644295340581e+01, 6.723854706303634998e+01, -6.008911815873547724e+00]

print ("Creating Environment")
env = RobotEnv(filename, image_files, desired_position)


state_dim = env.observation_space['position'].shape[0]
action_dim = env.action_space.n
hidden_size = 256

print ("Creating CQL Agent")
agent = CQLAgent(state_size=state_dim, action_size=action_dim, device="cpu")

print ("Start Trainning")
train_cql_agent(agent, env)