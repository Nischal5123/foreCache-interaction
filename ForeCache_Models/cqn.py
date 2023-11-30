

import environment2 as environment2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import pandas as pd
import random
import d3rlpy
eps=1e-35
from ray.rllib.examples.env.random_env import RandomEnv
from gym.spaces import Discrete, Box

config = {
            "framework": "tf",
            "explore": False,
            "env_config":{
                "action_space": Discrete(1),
                "observation_space": Discrete(1),
            }
        }

env = RandomEnv(config["env_config"])
def get_state_action_reward(user_list, algo, hyperparam_file,env_forecache):
    for u in user_list:
            env_forecache.process_data(u, 0)
    return np.array(env_forecache.mem_states) , np.array(env_forecache.mem_action), np.array(env_forecache.mem_reward)

def encode_state(state):
    if state=='Navigation':
        return 0
    elif state=='Foraging':
        return 1
    elif state=='Sensemaking':
        return 2
def encode_action(action):
    if action=='same':
        return 0
    elif action=='change':
        return 1
    elif action=='changeout':
        return 2
def decode_action(action):
    if action==0:
        return 'same'
    elif action==1:
        return 'change'
    elif action==2:
        return 'changeout'
def decode_state(state):
    if state==0:
        return 'Navigation'
    elif state==1:
        return 'Foraging'
    elif state==2:
        return 'Sensemaking'

if __name__ == "__main__":
    env_forecache = environment2.environment2()
    user_list_2D = [env_forecache.user_list_2D[0]]
    state,action,reward=get_state_action_reward(user_list_2D, 'Greedy', 'sampled-hyperparameters-config.json',env_forecache)
    for i in range(len(state)):
        state[i]=encode_state(state[i])
        action[i]=encode_action(action[i])
    terminal=np.zeros(len(state))
    terminal[-1]=1
    dataset = d3rlpy.dataset.MDPDataset(
        observations=state,
        actions=action,
        rewards=reward,
        terminals=terminal,
    )
    print(env.step())









