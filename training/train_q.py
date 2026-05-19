import sys
import os

# Add parent directory to path to allow importing env and agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.warehouse_env import WarehouseEnv
from agents.q_learning_agent import QLearningAgent

def train(episodes=5000):
    env = WarehouseEnv()
    agent = QLearningAgent(action_space_size=len(env.action_space))
    
    metrics = {
        'rewards': [],
        'success_rate': [],
        'episode_lengths': []
    }
    
    successes = 0
    
    print("Starting Q-Learning training...")
    print(f"Training for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 200:  # Increased max steps
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if reward == 100:
                successes += 1
                
        agent.decay_epsilon()
        
        metrics['rewards'].append(total_reward)
        metrics['episode_lengths'].append(steps)
        metrics['success_rate'].append(successes / (episode + 1))
        
        if (episode + 1) % 500 == 0:
            avg_reward = sum(metrics['rewards'][-500:]) / min(500, len(metrics['rewards']))
            success_rate = successes / (episode + 1)
            print(f"Episode: {episode+1}/{episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Success Rate: {success_rate:.2%} | Successes: {successes}")
            
    # Save the trained model
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'q_table.pkl')
    agent.save(checkpoint_path)
    print(f"\nTraining finished!")
    print(f"Final Success Rate: {successes / episodes:.2%}")
    print(f"Total Successes: {successes}/{episodes}")
    print(f"Q-table saved to {checkpoint_path}")
    
    # Generate plots
    from utils.visualization import plot_rewards, plot_success_rate
    plot_rewards(metrics, title="Q-Learning Training Rewards", filename="q_learning_rewards.png")
    plot_success_rate(metrics, title="Q-Learning Success Rate", filename="q_learning_success.png")
    print("Training plots saved to plots/ directory")
    
    return metrics

if __name__ == "__main__":
    train(episodes=5000)
