import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import ast
import random
from collections import deque


class environment_vizrec:
    def __init__(self):
        """
        Constructor for the environment class.

        Initializes required variables and stores data in memory.
        """
        self.user_list_movies_p1 = np.sort(glob.glob("data/zheng/processed_interactions_p1/*"))
        self.user_list_movies_p2 = np.sort(glob.glob("data/zheng/processed_interactions_p2/*"))
        self.user_list_movies_p3 = np.sort(glob.glob("data/zheng/processed_interactions_p3/*"))
        self.user_list_movies_p4 = np.sort(glob.glob("data/zheng/processed_interactions_p4/*"))
        self.user_list_birdstrikes_p1 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p1/*"))
        self.user_list_birdstrikes_p2 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p2/*"))
        self.user_list_birdstrikes_p3 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p3/*"))
        self.user_list_birdstrikes_p4 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p4/*"))


        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask

        # # List of valid actions and states
        # self.valid_actions = ['same','modify-x','modify-y','modify-z','modify-x-y','modify-y-z','modify-x-z','modify-x-y-z']
        #self.valid_actions = ['same', 'modify']
        self.valid_actions = ['same', 'modify-1', 'modify-2','modify-3']
        # self.valid_states = ['Task_Panel','Related_View','Top_Panel','Data_View']


        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.mem_roi = []
        self.threshold = 0
        self.prev_state = None

    def reset(self, all=False, test=False):
        """
        Reset the variables used for tracking position of the agents.

        :param all: If true, reset all variables.
        :param test: If true, set the current step to threshold value.
        :return: Current state.
        """
        if test:
            self.steps = self.threshold
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            self.mem_roi = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s

    def process_data(self, filename, thres):
        """
        Processes the data from the given file and stores it in the object memory.

        Inputs:
        - filename (str): path of the data file.
        - thres (float): threshold value to determine the end of the episode.

        Returns: None.
        """
        df = pd.read_csv(filename)
        cnt_inter = 0
        for index, row in df.iterrows():
            cur_state = row['State']
            action = row['Action']
            reward = row['Reward']
            self.mem_states.append(cur_state)
            self.mem_reward.append(reward)
            self.mem_action.append(action)
            cnt_inter += 1
        #scale reward min-max
        self.mem_reward = (self.mem_reward - np.min(self.mem_reward)) / (np.max(self.mem_reward) - np.min(self.mem_reward))


        self.threshold = round(cnt_inter * thres,0)

    def cur_inter(self, steps):
        """
        Returns the current state, reward, and action from the object memory.

        Inputs:
        - steps (int): current step of the episode.

        Returns:
        - (str): current state.
        - (float): current reward.
        - (str): current action.
        """
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        """
        Returns whether the next step exists and its index.

        Inputs: None.

        Returns:
        - (bool): True if the next step does not exist, False otherwise.
        - (int): index of the next step.
        """
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test=False):
        """
        Takes a step in the episode.

        Inputs:
        - test (bool): whether to take a test step.

        Returns: None.
        """
        if test:
            if len(self.mem_states) > self.steps + 1:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0
        else:
            if self.threshold > self.steps + 1:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0



    def step(self, cur_state, act_arg, test):
        # Get the current state, reward, and action at the current step
        _, cur_reward, cur_action = self.cur_inter(self.steps)

        # Peek at the next step to get the next state, reward, and action
        _, temp_step = self.peek_next_step()
        next_state, next_reward, next_action = self.cur_inter(temp_step)

        predicted_action = self.valid_actions[act_arg]

        # Initialize top_reward to 0
        top_reward = 0

        # If the predicted action is the same as the current action,
        if predicted_action == cur_action:
            prediction = 1
        else:
            # Otherwise, set prediction to 0 and cur_reward to 0
            prediction = 0
            cur_reward = 0

        # Set top_reward to cur_reward
        top_reward = cur_reward

        # Take the step action
        self.take_step_action(test)

        # Return the predicted next state, current reward, done status, prediction, and top reward
        return next_state, cur_reward, self.done, prediction, top_reward

    def get_user_list(self,dataset,task):
        if dataset == 'movies':
            if task == 'p1':
                return self.user_list_movies_p1
            elif task == 'p2':
                return self.user_list_movies_p2
            elif task == 'p3':
                return self.user_list_movies_p3
            elif task == 'p4':
                return self.user_list_movies_p4
        elif dataset == 'birdstrikes':
            if task == 'p1':
                return self.user_list_birdstrikes_p1
            elif task == 'p2':
                return self.user_list_birdstrikes_p2
            elif task == 'p3':
                return self.user_list_birdstrikes_p3
            elif task == 'p4':
                return self.user_list_birdstrikes_p4

if __name__ == "__main__":
    env = environment_vizrec()
