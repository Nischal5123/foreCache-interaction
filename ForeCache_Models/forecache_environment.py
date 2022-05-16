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



class environment3(Env):

    #ForeCache Data Exploration Environment

    def __init__(self):
        super(environment3, self).__init__()
        self.user_list = glob.glob('taskname_ndsi-2d-task_*')
        # This variable will be used to track the current position of the user agent.
        self.steps = 0

        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.data_length = 0  #training data length derived from threshold * train file length


        self.action_space= Discrete(2)   # only 2 actions 0:same 1:change
        self.valid_actions = [0, 1]  # ["same", "change"]


        self.state= None
        self.prev_state = None
        self.observation_space= Discrete(3)    #three states Sensemaking=0, Foraging=1, Navigation=2




    def step(self, action):
        state, reward, action = self.cur_inter(self.steps)
        #here action returned is ground action might use for accuracy later


        #parsed all of the training data , where data_length is max training data
        if self.steps == round(self.data_length):
            done=True
        else:
            done=False
        info={"Step":self.steps}

        self.steps+=1
        return state , reward, done ,info

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def process_data(self, filename,threshold):
        df = pd.read_csv(filename)
        self.prev_state = None
        self.data_length=len(df)*threshold
        cnt_inter = 0
        for index, row in df.iterrows():
            cur_state =(row['State'])

            if cur_state not in ('Foraging', 'Navigation', 'Sensemaking'):
                continue

            #assign numbers to state########################################## clean this out later
            if cur_state == 'Sensemaking':
                cur_state = 0
            if cur_state == 'Foraging':
                cur_state = 1
            if cur_state == 'Navigation':
                cur_state = 2
            #################################################################

            if self.prev_state == cur_state:
                action = "same"
            else:
                action = "change"
            self.mem_states.append(cur_state)
            self.mem_reward.append(row['NDSI'])
            self.mem_action.append(action)
            cnt_inter += 1
            self.prev_state = cur_state

    def reset(self, all=False, test=False):    #all clear per user  #default arguments after each episode
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
            print("start {}".format(self.steps))
            # pdb.set_trace()
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s

if __name__ == "__main__":
    env = environment3()
    users = env.user_list
    threshold=0.8
    env.process_data(users[0],threshold)
    episodes = 1  # 20 episodes
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))