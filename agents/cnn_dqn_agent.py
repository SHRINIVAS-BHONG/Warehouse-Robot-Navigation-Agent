import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

# Configure GPU settings
def configure_gpu():
    """Configure GPU for TensorFlow training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"SUCCESS: GPU detected and configured: {len(gpus)} GPU(s) available")
            print(f"  GPU devices: {[gpu.name for gpu in gpus]}")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("WARNING: No GPU detected. Training will use CPU.")
        print("  For GPU support on Windows:")
        print("  - Install tensorflow-directml: pip install tensorflow-directml")
        print("  - Or use WSL2 with CUDA-enabled TensorFlow")
        return False

# Configure GPU on module import
configure_gpu()

class CNNDQNAgent:
    """Phase 8 (Step 33): CNN-Based Vision Navigation.
    Uses a CNN to process the 5x5 partial observability grid,
    concatenated with a vector for the compass & battery.
    """
    def __init__(self, action_space_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_network()

    def _build_model(self, learning_rate):
        # Image input: 5x5 grid with 1 channel
        img_input = Input(shape=(5, 5, 1), name='grid_input')
        # Vector input: dx, dy, battery
        vec_input = Input(shape=(3,), name='sensor_input')
        
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(img_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = Flatten()(x)
        
        merged = Concatenate()([x, vec_input])
        
        dense = Dense(128, activation='relu')(merged)
        dense = Dense(64, activation='relu')(dense)
        output = Dense(self.action_space_size, activation='linear')(dense)
        
        model = tf.keras.Model(inputs=[img_input, vec_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def _split_state(self, state):
        img = state[:25].reshape(5, 5, 1)
        vec = state[25:]
        return img, vec

    def _split_batch(self, states):
        imgs = np.array([s[:25].reshape(5, 5, 1) for s in states])
        vecs = np.array([s[25:] for s in states])
        return [imgs, vecs]

    def get_action(self, state, explore=True):
        if explore and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        img, vec = self._split_state(state)
        img_expanded = np.expand_dims(img, axis=0).astype(np.float32)
        vec_expanded = np.expand_dims(vec, axis=0).astype(np.float32)
        q_values = self.model([img_expanded, vec_expanded], training=False).numpy()
        return np.argmax(q_values[0])

    def train_on_batch(self, states, actions, rewards, next_states, dones):
        states_split = self._split_batch(states)
        next_states_split = self._split_batch(next_states)
        
        states_tensors = [tf.convert_to_tensor(states_split[0], dtype=tf.float32), tf.convert_to_tensor(states_split[1], dtype=tf.float32)]
        next_states_tensors = [tf.convert_to_tensor(next_states_split[0], dtype=tf.float32), tf.convert_to_tensor(next_states_split[1], dtype=tf.float32)]
        
        target_q = self.model(states_tensors, training=False).numpy()
        future_q = self.target_model(next_states_tensors, training=False).numpy()
        
        for i in range(len(states)):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(future_q[i])
                
        self.model.fit(states_split, target_q, batch_size=len(states), epochs=1, verbose=0)

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
