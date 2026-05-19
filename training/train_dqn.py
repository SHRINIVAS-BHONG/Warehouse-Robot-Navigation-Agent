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
    # Improved hyperparameters
    agent = DQNAgent(
        state_shape=state_shape, 
        action_space_size=len(env.action_space),
        epsilon=1.0,
        epsilon_decay=0.9995,  # Slower decay for better exploration
        epsilon_min=0.05,  # Higher minimum for continued exploration
        learning_rate=0.0005  # Lower learning rate for stability
    )
    buffer = ReplayBuffer(capacity=10000)
    
    # Warm-start replay buffer with BFS expert trajectories
    from utils.expert_helper import get_shortest_path, get_action_from_states
    
    print("Warm-starting replay buffer with BFS expert trajectories...")
    path = get_shortest_path(env.grid_size, [0, 0], [9, 9], env.obstacles)
    if path:
        for _ in range(100):  # Increased from 50
            state, _ = env.reset()
            for step_idx in range(len(path) - 1):
                action = get_action_from_states(path[step_idx], path[step_idx+1])
                next_state = np.array(path[step_idx+1])
                reward = 100.0 if step_idx == len(path) - 2 else 1.0
                done = (step_idx == len(path) - 2)
                buffer.add(state, action, reward, next_state, done)
                state = next_state
        print(f"Replay buffer size after warm-start: {buffer.size()}")
        
        print("Pre-training DQN on expert buffer...")
        for _ in range(300):  # Increased from 150
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            agent.train_on_batch(states, actions, rewards, next_states, dones)
        print("Expert pre-training finished!")
    
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
        
        while not done and steps < 200:  # Increased from 100
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train more frequently
            if steps % 2 == 0 and buffer.size() >= batch_size:  # Changed from 4 to 2
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                agent.train_on_batch(states, actions, rewards, next_states, dones)
                
            if reward == 100:
                successes += 1
                
        agent.decay_epsilon()
        
        # Update target network more frequently
        if episode % 5 == 0:  # Changed from 10 to 5
            agent.update_target_network()
            
        metrics['rewards'].append(total_reward)
        metrics['episode_lengths'].append(steps)
        metrics['success_rate'].append(successes / (episode + 1))
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(metrics['rewards'][-10:]) / 10
            success_rate = successes / (episode + 1)
            print(f"Episode: {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Success Rate: {success_rate:.2%} | Successes: {successes}")
            
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'dqn_model.keras')
    agent.save(checkpoint_path)
    final_success_rate = successes / episodes
    print(f"\nTraining finished!")
    print(f"Final Success Rate: {final_success_rate:.2%}")
    print(f"Total Successes: {successes}/{episodes}")
    print(f"DQN model saved to {checkpoint_path}")
    return metrics

if __name__ == "__main__":
    train_dqn(episodes=200)  # Increased for better training
