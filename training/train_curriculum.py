import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.partial_obs_env import PartialObsWarehouseEnv
from agents.cnn_dqn_agent import CNNDQNAgent
from utils.replay_buffer import ReplayBuffer

def train_curriculum(batch_size=64):
    """Phase 8 (Step 32): Curriculum Learning.
    Trains the agent sequentially on progressively harder environments.
    Because PartialObsWarehouseEnv has a fixed state dimension (28 dim),
    the CNN agent can seamlessly transfer learned weights across grid sizes!
    """
    levels = [
        {"grid_size": 5, "num_obstacles": 2, "num_goals": 1, "max_battery": 50, "episodes": 100},
        {"grid_size": 8, "num_obstacles": 5, "num_goals": 2, "max_battery": 100, "episodes": 150},
        {"grid_size": 12, "num_obstacles": 15, "num_goals": 4, "max_battery": 250, "episodes": 200}
    ]
    
    agent = CNNDQNAgent(action_space_size=4)
    buffer = ReplayBuffer(capacity=20000)
    
    print("Starting Curriculum Learning with CNN DQN...")
    
    for level_idx, level in enumerate(levels):
        print(f"\n=== LEVEL {level_idx + 1}: Grid {level['grid_size']}x{level['grid_size']} | Obstacles: {level['num_obstacles']} | Goals: {level['num_goals']} ===")
        
        env = PartialObsWarehouseEnv(
            grid_size=level['grid_size'],
            num_obstacles=level['num_obstacles'],
            num_goals=level['num_goals'],
            max_battery=level['max_battery']
        )
        
        episodes = level['episodes']
        successes = 0
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 200:
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                
                buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if buffer.size() >= batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    agent.train_on_batch(states, actions, rewards, next_states, dones)
                    
                if done and reward >= 50: # Final goal reached
                    successes += 1
                    
            agent.decay_epsilon()
            
            if episode % 10 == 0:
                agent.update_target_network()
                
            if (episode + 1) % 10 == 0:
                success_rate = successes / (episode + 1)
                print(f"Level {level_idx+1} - Episode: {episode+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Success Rate: {success_rate:.2%}")
                
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'curriculum_cnn_model.keras')
    agent.save(checkpoint_path)
    print(f"\nCurriculum Training finished! Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train_curriculum()
