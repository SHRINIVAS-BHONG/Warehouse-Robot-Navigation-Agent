"""
Random Agent Demo
Demonstrates the warehouse environment with a random agent
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.warehouse_env import WarehouseEnv
import random

def main():
    print("="*60)
    print("🤖 Random Agent Demo - Warehouse Navigation")
    print("="*60)
    print("\nThis demo shows a random agent navigating the warehouse.")
    print("The agent takes random actions without learning.\n")
    
    env = WarehouseEnv()
    state, info = env.reset()
    
    done = False
    steps = 0
    total_reward = 0
    
    print("Starting navigation...\n")
    env.render()
    
    while not done and steps < 50:
        # Random action
        action = random.randint(0, 3)
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action={action_names[action]}, Position={state}, Reward={reward:.1f}")
        env.render()
        time.sleep(0.2)  # Slow down for visibility
        
    print("\n" + "="*60)
    if reward == 100:
        print(f"✅ Goal reached in {steps} steps!")
    else:
        print(f"❌ Failed to reach goal in {steps} steps")
    print(f"Total Reward: {total_reward:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()