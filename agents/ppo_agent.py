import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PPOAgent:
    def __init__(self, state_dim, num_actions, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, lam=0.97):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.lam = lam
        self.num_actions = num_actions
        
        # Actor Network
        state_input = layers.Input(shape=(state_dim,))
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        logits = layers.Dense(num_actions)(x)
        self.actor = tf.keras.Model(inputs=state_input, outputs=logits)
        
        # Critic Network
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        value = layers.Dense(1)(x)
        self.critic = tf.keras.Model(inputs=state_input, outputs=value)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        logits = self.actor(state)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        # Adding a tiny epsilon to ensure probabilities sum to 1 strictly
        action_probs = action_probs / np.sum(action_probs)
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def discounted_cumulative_sums(self, x, discount):
        res = np.zeros_like(x, dtype=np.float32)
        res[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            res[t] = x[t] + discount * res[t+1]
        return res

    @tf.function
    def train_policy(self, states, actions, advantages, log_probs_old):
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            action_probs = tf.nn.softmax(logits)
            
            # Get probability of selected actions
            actions_onehot = tf.one_hot(actions, self.num_actions)
            log_probs = tf.reduce_sum(actions_onehot * tf.math.log(action_probs + 1e-10), axis=1)
            
            ratio = tf.exp(log_probs - log_probs_old)
            min_adv = tf.where(advantages > 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))
            
        actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        return policy_loss

    @tf.function
    def train_value(self, states, returns):
        with tf.GradientTape() as tape:
            values = self.critic(states)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)
            
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return value_loss
        
    def save(self, actor_path, critic_path):
        import os
        os.makedirs(os.path.dirname(actor_path), exist_ok=True)
        self.actor.save(actor_path)
        self.critic.save(critic_path)
