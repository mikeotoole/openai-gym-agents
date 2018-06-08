import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# An Agent that uses a deep Q-network. This is a deep neural network with
# reinforcement learning.
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # deque is a Queue Data structure where you can add and remove from both ends.
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95 # A hyperparameter. Discount rate for rewards.
        
        # When net starts playing. Explore at random. Explore vs. Exploit.
        # The closer to one this value is the more it will explore.
        # The closer to zero this value is the more it will exploit what is has learned.
        # Probably want this to always be 1.0.
        self.epsilon = 1.0

        # Each timestamp epsilon will decrease and explore less.
        # This one you may want to experiment with.
        self.epsilon_decay = 0.995

        # We always want to explore some. Probably a good set min.
        self.epsilon_min = 0.01
        
        self.learning_rate = 0.001 # How much net can learn each timestamp.
        
        self.model = self._build_model() 
    
    def _build_model(self):

        model = Sequential()

        # NEURAL NETWORK ARCHITECTURE
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        
        model.add(Dense(24, activation='relu'))
        
        # 'linear' because we don't want probabilities.
        model.add(Dense(self.action_size, activation='linear'))

        # COMPILE MODEL
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    # done is if we died or not.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    # Get what action the agent should take.
    def act(self, state):
        # When self.epsilon is large we want to just do a random action.
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        
        # Otherwise we use the state to predict what the best action is.
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 

    # Does memory replay to learn.
    def replay(self, batch_size): 
        
        # sample 32 from 2000
        minibatch = random.sample(self.memory, batch_size) 
        
        # Fit the model. 
        for state, action, reward, next_state, done in minibatch: 
            # If we died reward is known.
            target = reward # N.B.: if done
            if not done: 
                # Predict what reward will be given next state.
                # (maximum target Q based on future action a')
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) 
            target_f[0][action] = target
            
            # 
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # If our epsilon is larger then min then decay it some.
        # This results in the agent doing less and less random action.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
