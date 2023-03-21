from collections import deque
import torch                    # For creating the neural network
import numpy as np
import random


# Trainning Function
# Hyper Parameters adjust here
def train_cql_agent(agent, env, num_episodes=250, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, buffer_size=100000):
    replay_buffer = deque(maxlen=buffer_size)
    epsilon = epsilon_start
    rewards = []
    losses = []


    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_losses = []


        while not done:
            action = agent.get_action(state['position'], epsilon)
            next_state, reward, done, _ = env.step(action[0])

            transition = (torch.tensor(np.array(state['position']), dtype=torch.float32).unsqueeze(0),
                          torch.tensor(np.array([action]), dtype=torch.int64).unsqueeze(1),
                          torch.tensor(np.array([reward]), dtype=torch.float32).unsqueeze(1),
                          torch.tensor(np.array(next_state['position']), dtype=torch.float32).unsqueeze(0),
                          torch.tensor(np.array([done]), dtype=torch.float32).unsqueeze(1))


            replay_buffer.append(transition)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                experiences = random.sample(replay_buffer, batch_size)
                loss = agent.learn(experiences)
                episode_losses.append(loss)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards.append(total_reward)
        losses.append(np.mean(episode_losses))
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward} - Loss: {np.mean(loss)}")

    return rewards, losses
