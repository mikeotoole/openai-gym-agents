import gym
import numpy as np
import os 
from dnq_agent import DQNAgent

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

batch_size = 32
n_episodes = 1001

output_dir = 'model_output/cartpole/'

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

agent = DQNAgent(state_size, action_size) 

done = False

for e in range(n_episodes): # Main episode loop.
  state = env.reset() # Reset the gym env and get the initial state.
  state = np.reshape(state, [1, state_size])
  
  for time in range(5000): # TODO: How does max time of 200 relate to 5000?
    # Watch agent play game. Needs to be commented out if running in notebook.        
    env.render() # This will slowdown the runs.

    # Get the action the agent wants to take given the env state.
    action = agent.act(state)

    # Preform the action on the env and get the next_state and reward.
    # `done` will be True if it was Game Over.
    next_state, reward, done, _ = env.step(action) 
    
    reward = reward if not done else -10 # If Game was over set reward to -10.
    
    # TODO: Check what this does to next_state.
    next_state = np.reshape(next_state, [1, state_size])
    
    # Store what happened. Given a state and action what was the reward and
    # next_state.
    # TODO: If reward is -10 why do we need to remember done? Because "done"
    #       reward can change?
    agent.remember(state, action, reward, next_state, done) 

    state = next_state 
    
    if done: 
      print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, time, agent.epsilon))
      break 

    if len(agent.memory) > batch_size:
      agent.replay(batch_size)

    if e % 50 == 0:
      agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

  # saved agents can be loaded with agent.load("./path/filename.hdf5") 

