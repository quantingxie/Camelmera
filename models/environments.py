import numpy as np              # For numerical operations
import gym                      # For creating the custom gym environment
from gym import spaces          # For defining action and observation spaces
from torchvision import transforms # For image transformations
from PIL import Image           # For image processing
from utils import load_image_files




class RobotEnv(gym.Env):
    # Initialize the environment with necessary parameters
    def __init__(self, positions_file, image_files, desired_position):
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




