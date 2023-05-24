import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import forecache_environment

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

if __name__ == "__main__":
    env = forecache_environment.environment3()
    users = env.user_list
    threshold=0.8
    env.process_data(users[0],threshold)


    states = env.observation_space.n
    actions = env.action_space.n


    def build_model(states, actions):  #default states shape (1,)
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=(1,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model


    def build_agent(model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy,
                       nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn


    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)

    results = dqn.test(env, nb_episodes=150, visualize=False)
    print(np.mean(results.history['episode_reward']))