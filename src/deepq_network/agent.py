import collections
import keras.layers as layers
import keras.models as models
import keras.optimizers as optimizers
import numpy as np
import random

class DQNAgent:

    def __init__(self, n_state_features, n_actions, epsilon=1.0):
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = 0.999  # exploration decay
        self.epsilon_min = 0.001
        self.gamma = 1.0    # discount factor
        self.learning_rate = 0.0001
        self.memory = collections.deque(maxlen=20000)
        self.n_actions = n_actions
        self.n_state_features = n_state_features
        self.model = self.create_model()

    def create_model(self):
        """
        Build a simple Neural Network (3 hidden layers)
        which will be used as a Value Function Approximation.

        return:
            Keras NN model.
        """
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.n_state_features, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.n_actions, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def experience_replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminated in batch:
            if not terminated:
                # target = immediate reward + (discount factor * value of next state)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                # if it's a terminal state,
                # the value of this state equals to immediate reward
                target = reward
            # Predict value for a current state
            final_target = self.model.predict(state)
            # Update state-action value with a better value
            final_target[0][action] = target
            self.model.fit(state, final_target, epochs=1, verbose=0)
        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        """
        Perform an action given environment state.

        state:
            Discrete environment state (integer)

        return:
            Action to be performed (integer)
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        action_values = self.model.predict(state)
        best_action = np.argmax(action_values[0])
        return best_action

    def load(self, name):
        """
        Load saved Value Function Approximation weights.

        name:
            Model filename.
        """
        self.model.load_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        """
        Save Value Function Approximation weights.

        name:
            Model filename.
        """
        self.model.save_weights(name)
