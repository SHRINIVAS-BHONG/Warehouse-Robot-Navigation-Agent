import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.warehouse_env import WarehouseEnv
from agents.ppo_agent import PPOAgent

def train_ppo(epochs=50, steps_per_epoch=1000):
    env = WarehouseEnv()
    state_dim = 2
    num_actions = len(env.action_space)
    
    agent = PPOAgent(state_dim=state_dim, num_actions=num_actions)
    
    # Pre-train PPO using Behavioral Cloning on expert path
    from utils.expert_helper import get_shortest_path, get_action_from_states
    
    print("Pre-training PPO using Behavioral Cloning on expert path...")
    path = get_shortest_path(env.grid_size, [0, 0], [9, 9], env.obstacles)
    if path:
        expert_states = []
        expert_actions = []
        for step_idx in range(len(path) - 1):
            expert_states.append(path[step_idx])
            expert_actions.append(get_action_from_states(path[step_idx], path[step_idx+1]))
            
        expert_states = np.array(expert_states, dtype=np.float32)
        expert_actions = np.array(expert_actions, dtype=np.int32)
        
        # Train policy and value networks on expert data
        for _ in range(50):
            # Fit actor
            with tf.GradientTape() as tape:
                logits = agent.actor(expert_states, training=True)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        expert_actions, logits, from_logits=True
                    )
                )
            grads = tape.gradient(loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))
            
            # Fit critic
            returns = np.zeros(len(path) - 1, dtype=np.float32)
            discounted_sum = 0
            for idx in reversed(range(len(path) - 1)):
                reward = 100.0 if idx == len(path) - 2 else 1.0
                discounted_sum = reward + agent.gamma * discounted_sum
                returns[idx] = discounted_sum
                
            with tf.GradientTape() as tape:
                values = agent.critic(expert_states, training=True)
                loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(returns, dtype=tf.float32) - tf.squeeze(values)))
            grads = tape.gradient(loss, agent.critic.trainable_variables)
            agent.critic_optimizer.apply_gradients(zip(grads, agent.critic.trainable_variables))
            
        print("PPO expert pre-training finished!")
    
    # Buffers
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    values_buffer = []
    logprobs_buffer = []
    
    metrics = {
        'rewards': [],
        'success_rate': []
    }
    
    print("Starting PPO training...")
    
    for epoch in range(epochs):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        successes = 0
        episodes_in_epoch = 0
        
        # Collect trajectories
        for t in range(steps_per_epoch):
            action = agent.get_action(state)
            
            # Get value and logprob
            state_input = np.expand_dims(state, axis=0)
            value = agent.critic(state_input).numpy()[0][0]
            logits = agent.actor(state_input)
            action_probs = tf.nn.softmax(logits).numpy()[0]
            logprob = np.log(action_probs[action] + 1e-10)
            
            next_state, reward, done, _, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            
            states_buffer.append(state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            logprobs_buffer.append(logprob)
            
            state = next_state
            
            if done or t == steps_per_epoch - 1:
                # Bootstrap value
                if done:
                    last_val = 0
                else:
                    last_val = agent.critic(np.expand_dims(state, axis=0)).numpy()[0][0]
                    
                rewards_with_last = rewards_buffer + [last_val]
                values_with_last = values_buffer + [last_val]
                
                # GAE and Returns
                deltas = np.array(rewards_buffer) + agent.gamma * np.array(values_with_last[1:]) - np.array(values_buffer)
                advantages = agent.discounted_cumulative_sums(deltas, agent.gamma * agent.lam)
                returns = agent.discounted_cumulative_sums(rewards_with_last, agent.gamma)[:-1]
                
                # Normalize advantages
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
                
                # Train networks
                agent.train_policy(
                    tf.convert_to_tensor(states_buffer, dtype=tf.float32),
                    tf.convert_to_tensor(actions_buffer, dtype=tf.int32),
                    tf.convert_to_tensor(advantages, dtype=tf.float32),
                    tf.convert_to_tensor(logprobs_buffer, dtype=tf.float32)
                )
                agent.train_value(
                    tf.convert_to_tensor(states_buffer, dtype=tf.float32),
                    tf.convert_to_tensor(returns, dtype=tf.float32)
                )
                
                # Clear buffers
                states_buffer.clear()
                actions_buffer.clear()
                rewards_buffer.clear()
                values_buffer.clear()
                logprobs_buffer.clear()
                
                if reward == 100:
                    successes += 1
                
                metrics['rewards'].append(episode_return)
                episodes_in_epoch += 1
                
                state, _ = env.reset()
                episode_return = 0
                episode_length = 0
                
        # Epoch metrics
        avg_reward = sum(metrics['rewards'][-episodes_in_epoch:]) / episodes_in_epoch if episodes_in_epoch > 0 else 0
        success_rate = successes / episodes_in_epoch if episodes_in_epoch > 0 else 0
        metrics['success_rate'].append(success_rate)
        
        print(f"Epoch {epoch+1}/{epochs} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2%}")
        
    actor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'ppo_actor.keras')
    critic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'ppo_critic.keras')
    agent.save(actor_path, critic_path)
    print(f"Training finished and PPO models saved to {os.path.dirname(actor_path)}")
    return metrics

if __name__ == "__main__":
    train_ppo(epochs=10, steps_per_epoch=100)  # Reduced for faster training
