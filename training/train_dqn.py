import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def train_dqn(episodes=500, batch_size=64):
    env = WarehouseEnv()
    
    state_shape = (2,) # (x, y)
    agent = DQNAgent(state_shape=state_shape, action_space_size=len(env.action_space))
    buffer = ReplayBuffer(capacity=10000)
    
    successes = 0
    metrics = {
        'rewards': [],
        'success_rate': [],
        'episode_lengths': []
    }
    
    print("Starting DQN training...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                agent.train_on_batch(states, actions, rewards, next_states, dones)
                
            if reward == 100:
                successes += 1
                
        agent.decay_epsilon()
        
        if episode % 10 == 0:
            agent.update_target_network()
            
        metrics['rewards'].append(total_reward)
        metrics['episode_lengths'].append(steps)
        metrics['success_rate'].append(successes / (episode + 1))
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(metrics['rewards'][-10:]) / 10
            success_rate = successes / (episode + 1)
            print(f"Episode: {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Success Rate: {success_rate:.2%}")
            
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'dqn_model.keras')
    agent.save(checkpoint_path)
    print(f"Training finished and DQN model saved to {checkpoint_path}")
    return metrics

if __name__ == "__main__":
    train_dqn(episodes=200) # Shortened for demonstration
