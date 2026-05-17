import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.warehouse_env import WarehouseEnv
from agents.q_learning_agent import QLearningAgent
from utils.visualization import plot_heatmap

def evaluate():
    env = WarehouseEnv()
    agent = QLearningAgent(action_space_size=len(env.action_space), epsilon=0) # No exploration
    
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'q_table.pkl')
    try:
        agent.load(checkpoint_path)
        print(f"Loaded Q-table from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Initialize visitation matrix for heatmap
    visitation_matrix = np.zeros((env.grid_size, env.grid_size))
    visitation_matrix[state[1], state[0]] += 1
    
    print("Starting evaluation...")
    env.render()
    
    while not done and steps < 50:
        action = agent.get_action(state, explore=False)
        state, reward, done, _, _ = env.step(action)
        
        # Track visited coordinate
        visitation_matrix[state[1], state[0]] += 1
        
        total_reward += reward
        steps += 1
        
        print(f"Step: {steps}, Action: {action}, Reward: {reward}")
        env.render()
        time.sleep(0.1) # slightly faster evaluation
        
    if reward >= 100:
        print(f"\nGoal reached in {steps} steps! Total Reward: {total_reward}")
    else:
        print(f"\nFailed to reach goal. Total Reward: {total_reward}")
        
    # Generate and save heatmap
    plot_heatmap(visitation_matrix, title="Agent Path Heatmap", filename="evaluation_heatmap.png")
    print("Saved Path Heatmap to plots/evaluation_heatmap.png")

if __name__ == "__main__":
    evaluate()
