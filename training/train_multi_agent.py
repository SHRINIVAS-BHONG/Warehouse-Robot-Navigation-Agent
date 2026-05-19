import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_agent_env import MultiAgentWarehouseEnv
from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer

def train_multi_agent(episodes=200, batch_size=64):
    """Phase 8 (Step 31): Multi-Agent Training.
    Trains two independent decentralized DQN agents in a shared environment.
    """
    env = MultiAgentWarehouseEnv()
    
    # State shape is 8 (r1_x, r1_y, r2_x, r2_y, g1_x, g1_y, g2_x, g2_y)
    state_shape = (env.state_dim,)
    
    # Independent learners
    agent1 = DQNAgent(state_shape=state_shape, action_space_size=len(env.action_space))
    agent2 = DQNAgent(state_shape=state_shape, action_space_size=len(env.action_space))
    
    buffer1 = ReplayBuffer(capacity=10000)
    buffer2 = ReplayBuffer(capacity=10000)
    
    print("Starting Multi-Agent Decentralized DQN Training...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_r1 = 0
        total_r2 = 0
        
        while not done and steps < 100:
            a1 = agent1.get_action(state)
            a2 = agent2.get_action(state)
            
            next_state, rewards, done, _, _ = env.step((a1, a2))
            r1, r2 = rewards
            
            buffer1.add(state, a1, r1, next_state, done)
            buffer2.add(state, a2, r2, next_state, done)
            
            state = next_state
            total_r1 += r1
            total_r2 += r2
            steps += 1
            
            if steps % 4 == 0 and buffer1.size() >= batch_size:
                s1, ac1, rew1, ns1, d1 = buffer1.sample(batch_size)
                agent1.train_on_batch(s1, ac1, rew1, ns1, d1)
                
            if steps % 4 == 0 and buffer2.size() >= batch_size:
                s2, ac2, rew2, ns2, d2 = buffer2.sample(batch_size)
                agent2.train_on_batch(s2, ac2, rew2, ns2, d2)
                
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode+1}/{episodes} | Avg R1: {total_r1:.1f} | Avg R2: {total_r2:.1f} | Epsilon: {agent1.epsilon:.4f}")
            
    # Save both agents
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    agent1.save(os.path.join(checkpoints_dir, 'multi_agent_r1.keras'))
    agent2.save(os.path.join(checkpoints_dir, 'multi_agent_r2.keras'))
    print(f"\nTraining finished! Multi-agent models saved to {checkpoints_dir}")

if __name__ == "__main__":
    train_multi_agent(episodes=40)  # Reduced for faster training
