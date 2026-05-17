import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_shape, action_space_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_network()

    def _build_model(self, learning_rate):
        model = Sequential([
            Input(shape=self.state_shape),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_space_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, explore=True):
        if explore and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        state_expanded = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_expanded, verbose=0)
        return np.argmax(q_values[0])

    def train_on_batch(self, states, actions, rewards, next_states, dones):
        target_q = self.model.predict(states, verbose=0)
        future_q = self.target_model.predict(next_states, verbose=0)
        
        for i in range(len(states)):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(future_q[i])
                
        self.model.fit(states, target_q, batch_size=len(states), epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            self.update_target_network()
        else:
            print(f"Warning: Model {filepath} not found.")
