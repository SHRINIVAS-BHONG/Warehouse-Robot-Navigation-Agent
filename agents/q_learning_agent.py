import numpy as np
import random
import pickle
import os

class QLearningAgent:
    def __init__(self, action_space_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
        return self.q_table[state][action]

    def get_action(self, state, explore=True):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_size)
            
        # Epsilon-Greedy Exploration
        if explore and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: choose best known action
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space_size)
            
        # Bellman Q-Update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.get_q_value(state, action)
        
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            print(f"Warning: File {filepath} not found.")
