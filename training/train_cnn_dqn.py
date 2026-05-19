import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.partial_obs_env import PartialObsWarehouseEnv
from agents.cnn_dqn_agent import CNNDQNAgent
from utils.replay_buffer import ReplayBuffer

def train_cnn_dqn(episodes=200, batch_size=64):
    env = PartialObsWarehouseEnv()
    
    agent = CNNDQNAgent(action_space_size=len(env.action_space))
    buffer = ReplayBuffer(capacity=10000)
    
    successes = 0
    metrics = {
        'rewards': [],
        'success_rate': [],
        'episode_lengths': []
    }
    
    print("Starting CNN DQN training on Partial Observability Environment...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Expert guidance prob
        import random
        expert_prob = max(0.0, 0.8 - (episode / 40.0))
        from utils.expert_helper import get_shortest_path, get_action_from_states
        
        while not done and steps < 150: # Allow more steps due to larger map and goals
            expert_action = None
            if env.current_goal_idx < len(env.goals):
                path = get_shortest_path(env.grid_size, env.robot_pos, env.goals[env.current_goal_idx], env.obstacles)
                if path and len(path) > 1:
                    expert_action = get_action_from_states(env.robot_pos, path[1])
                    
            action = agent.get_action(state)
            if expert_action is not None and random.random() < expert_prob:
                action = expert_action
                
            next_state, reward, done, _, _ = env.step(action)
            
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % 4 == 0 and buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                agent.train_on_batch(states, actions, rewards, next_states, dones)
                
            if done and reward >= 50: # Final goal reached
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
            
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'cnn_dqn_model.keras')
    agent.save(checkpoint_path)
    print(f"Training finished and CNN DQN model saved to {checkpoint_path}")

if __name__ == "__main__":
    train_cnn_dqn(episodes=30)  # Reduced for faster training
