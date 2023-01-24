# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
from random import sample

class stationarity_test:
    def __init__(self):
        path = os.getcwd()
        self.user_list_faa = glob.glob("D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\WindowSize\\*.csv")
        self.user_list_faa_feedback = glob.glob("D:\\imMens Learning\\Faa_neew\\new_mdp\\*-0ms.xlsx")

        self.vizs = defaultdict(list)
        self.feed_time_reward = defaultdict(list)

    def window_size(self, filename):
        for names in filename:
            # print(names)
            file = open(names, 'r')
            csv_reader = csv.reader(file)
            prev_time = None
            prev_vis = None
            for interaction in csv_reader:
                time = interaction[0].split(':')
                cur_time = 60 * int(time[0]) + int(time[1])
                cur_vis = interaction[2]
                # print(time_second, vis)
                if prev_vis != None and prev_vis != cur_vis:
                    time_spent = cur_time - prev_time
                    self.vizs[cur_vis].append(time_spent)
                prev_vis = cur_vis
                prev_time = cur_time
        # print(self.vizs)
        for keys in self.vizs:
            sum = 0
            num = 0
            for items in self.vizs[keys]:
                if items > 5:
                    sum += items
                    num += 1
            print(keys, round(sum / num, 2))

    def accumulated_reward(self, fnames):
        df = pd.read_excel(fnames, sheet_name="Sheet3", usecols="A:F")
        for index, row in df.iterrows():
            time = row['time'].split(':')
            feedback_time = time[1] * 60 + time[2]





if __name__ == "__main__":
    obj = stationarity_test()

    # Uncomment for checking the window size
    # rand_num = random.sample(range(0, len(obj.user_list_faa)), 4)
    # fname = []
    # for i in rand_num:
    #     fname.append(obj.user_list_faa[i])
    # obj.window_size(fname)


    for users in obj.user_list_faa_feedback:
        print(users)




# Window Sizes
#  bar-4 45.17
#  bar-2 158.43
#  scatterplot-0-1 84.15
#  hist-3 76.14