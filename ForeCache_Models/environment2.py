import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
import random
from collections import deque

class environment2:
    def __init__(self):
        self.user_list_2D = np.sort(glob.glob('data/NDSI-2D/taskname_ndsi-2d-task_*'))
        self.user_list_3D = np.sort(glob.glob('data/NDSI-3D/taskname_ndsi-3d-task_*'))

        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.valid_actions = ['same','change']
        self.valid_states = ['Foraging', 'Navigation', 'Sensemaking']
        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.mem_roi=[]
        self.threshold = 0
        self.prev_state = None

    def reset(self, all=False, test = False):
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
           # print("start {}".format(self.steps))
            # pdb.set_trace()
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

    def get_state(self, state):

        return state

    # Optimization is not the priority right now
    # Returns all interactions for a specific user and subtask i.e. user 'P1', subtask '1.txt'
    def process_data(self, filename, thres):
        # df = pd.read_excel(filename, sheet_name= "Sheet1", usecols="B:D")
        df = pd.read_csv(filename)

        self.prev_state = 'Foraging'
        cnt_inter = 0
        roi_subset = []
        subset = 1
        for index, row in df.iterrows():
            # pdb.set_trace()
            # print("here {} end\n".format(cnt_inter))
            cur_state = self.get_state(row['State'])
            if cur_state not in ('Foraging','Navigation', 'Sensemaking'):
                continue
            if self.prev_state == cur_state:
                action = "same"
            else:
                action = "change"
            if cur_state == 'Sensemaking':
                if (index < (len(df) - 1)) and df['State'][index + 1] != 'Sensemaking':
                    roi_subset.append(subset)
                    subset = subset + 1
                    row['NDSI']+=5


                else:
                    roi_subset.append(subset)
            else:
                roi_subset.append(subset)

            self.mem_states.append(cur_state)
            self.mem_reward.append(row['NDSI']*row['ZoomLevel'])
            self.mem_action.append(action)
            cnt_inter += 1
            self.prev_state=cur_state
        self.mem_action = self.mem_action[1:] +['same']
        self.mem_roi=roi_subset
        self.threshold = int(cnt_inter * thres)
       # print("{} {}\n".format(len(self.mem_states), self.threshold))

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test = False):
        # pdb.set_trace()
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

    # act_arg = action argument refers to action number
    def step(self, cur_state, act_arg, test):
        _, cur_reward, cur_action = self.cur_inter(self.steps)
        _, temp_step = self.peek_next_step()
        next_state, next_reward, next_action = self.cur_inter(temp_step)
        # pdb.set_trace()
        predicted_action=self.valid_actions[act_arg]
        if predicted_action == cur_action:
            prediction = 1

        else:
            prediction = 0
            cur_reward = 0


        self.take_step_action(test)
        return next_state, cur_reward, self.done, prediction


if __name__ == "__main__":
    env = environment2()
    users = env.user_list_2D
    env.process_data(users[0],0.5)
    print(users)
