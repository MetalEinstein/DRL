import random
import gym
import numpy as np
from collections import deque
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import os
import custom_env

env = gym.make('Custom_Box-Env-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

batch_size = 32
n_episodes = 3000  # Number of games we want agent to play



output_dir = 'model_output/weights/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(
            maxlen=2000)  # double-ended queue; acts like list, but elements can be added/removed from either end

        self.gamma = 0.95  # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0  # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.99995  # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01  # minimum amount of random exploration permitted
        self.learning_rate = 0.001  # rate at which NN adjusts models parameters via SGD to reduce cost
        self.model = self._build_model()  # private method

    # Our neural network model is used to estimate the Q-value of an action given a specific state
    def _build_model(self):
        # We build our neural network in a sequential manner. First layer defined is the input layer,
        # last layer is the output layer and everything in between is hidden layers
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # 1st layer; states as input
        model.add(Dense(24, activation='relu'))  # Hidden layer
        model.add(Dense(self.action_size, activation='linear'))  # 4 actions, so 4 output neurons: 0, 1, 2, 3 (L/R/U/D)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))  # Mean-Squared Error for calculating loss
        return model

    # We remember a certain memory to store in the deque
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))  # list of previous experiences, enabling re-training later

    # We define how to balance exploration and exploitation using epsilon-greedy
    def act(self, state):
        if np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)  # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0])  # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size):  # method that trains NN with experiences sampled from memory
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        state = np.zeros((batch_size, self.state_size))
        next_state = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        
        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Load weights for further training
    def load(self, name):
        self.model.load_weights(name)

    # Save weights
    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size)
game_done = False
reward_list = []

for episodes in range(n_episodes):
    game_done = False
    state = env.reset()
    state = np.reshape(state, [1, state_size])  # Transpose the array to fit with the network
    reward_list.clear()

    while not game_done:
        env.render()
        action = agent.act(state)  # Action will be either 0, 1, 2 or 3 for left, right, up, down respectively
        next_state, reward, game_done = env.step(action)  # We send an action to the environment and receives a new observation/state
        next_state = np.reshape(next_state, [1, state_size])

        # We add the reward to a list in order to find the accumulated reward for this episode
        reward_list.append(reward)

        # Add a memory to be replayed later
        agent.remember(state, action, reward, next_state, game_done)

        # We store the next_state as the previous state for the next loop
        state = next_state

        # If the agent is done with an episode
        if game_done:
            print("episode: {}/{}, score: {}, e: {:.2}"  # print the episode's score and agent's epsilon
                  .format(episodes, n_episodes, sum(reward_list), agent.epsilon))
            break  # exit loop

        # When our memory batch is full we start replaying them
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # train the agent by replaying the experiences of the episode

env.close()



