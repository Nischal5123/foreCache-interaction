import csv
import pdb
import glob
import random

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import rel_entr


class integrate:
    def __init__(self):
        self.raw_files = glob.glob("D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\RawInteractions\\*.csv")
        self.excel_files = glob.glob("D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\FeedbackLog\\*.xlsx")
        self.path = 'D:\\imMens Learning\\stationarity_test\\KLDivergenceTest\\Merged\\'
        self.vizs = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1']
        self.cum_rewards = defaultdict(list)

    def debug(self, user, raw_data, feedback_data):
        print(user)
        print("#######RAW#########")
        for idx in range(len(raw_data)):
            print(idx, raw_data[idx][0], raw_data[idx][1], raw_data[idx][2])
        print("#######EXCEL#########")
        for idx in range(len(feedback_data)):
            print(idx, feedback_data[idx][0], feedback_data[idx][2], feedback_data[idx][3])

        print(user)
        for v in self.cum_rewards:
            print(v)
            for keys in self.cum_rewards[v]:
                print(keys)

    def get_files(self):
        for_now = ['p3', 'p7', 'p11']
        for raw_fname in self.raw_files:
            user = Path(raw_fname).stem.split('-')[0]
            if user not in for_now:
                continue
            excel_fname = [string for string in self.excel_files if user in string][0]
            self.cum_rewards.clear()
            self.merge(user, raw_fname, excel_fname)

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            if row['State'] == "None": #When reading from excel do not consider states = None
                continue
            data.append([seconds, row['proposition'], row['Reward'], row['State']])
        return data

    def raw_to_memory(self, csv_reader):
        next(csv_reader)
        data = []
        for lines in csv_reader:
            time = lines[0].split(":")
            mm = int(time[1])
            ss = int(time[2])
            seconds = mm * 60 + ss
            data.append([seconds, lines[1], lines[2]])
        return data

    def check_cur_viz(self, raw_data, cur_time, target_viz):
        stime = cur_time - 10
        etime = cur_time
        for idx in range(len(raw_data) - 1):
            if raw_data[idx][0] <= etime <= raw_data[idx + 1][0] or raw_data[idx][0] <= stime <= raw_data[idx + 1][0]:
                if raw_data[idx][2] == target_viz:
                    return True
                else:
                    return False

    def check_cur_reward(self, feedback_data, cur_time):
        stime = cur_time - 10
        etime = cur_time
        reward = 0
        for idx in range(len(feedback_data)):
            if stime <= feedback_data[idx][0] <= etime:
                reward += feedback_data[idx][2]
            elif feedback_data[idx][0] > etime:
                break
        return reward

    def find_runtime(self, raw_data, feedback_data):
        l1 = len(raw_data)
        l2 = len(feedback_data)
        _max = max(int(raw_data[l1 - 1][0]), int(feedback_data[l2 - 1][0]))
        return _max

    #Used for merging the raw interaction (reformed) files with the Excel feedback files
    def merge(self, user, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet3", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)

        runtime = self.find_runtime(raw_data, feedback_data)

        for v in self.vizs:
            cur_time = 0
            cum_reward = 0
            while cur_time < runtime:
                cur_time += 10
                if self.check_cur_viz(raw_data, cur_time, v):
                    cum_reward += self.check_cur_reward(feedback_data, cur_time)
                    # pdb.set_trace()
                self.cum_rewards[v].append((cur_time, cum_reward))
        # self.plot_graph(user)
        self.stationarity_testing(user, runtime)

    def get_windows(self, runtime):
        # Randomly picking two windows:
        # move_start = (runtime % 90)
        move_start = 50
        windows = int(runtime / 90)
        # print("debug {} {} {}".format(runtime, move_start, windows))
        # for i in range(10):
        a1, a2 = random.sample(range(1, windows), 2)
        w1 = []
        s1 = move_start + a1 * 90
        for i in range(1, 9):
            w1.append(s1)
            s1 += 10

        w2 = []
        s2 = move_start + a2 * 90
        for i in range(1, 9):
            w2.append(s2)
            s2 += 10
        window1 = defaultdict(list)
        window2 = defaultdict(list)
        for v in self.cum_rewards:
            w_idx = 0
            for idx in range(len(self.cum_rewards[v])):
                if self.cum_rewards[v][idx][0] == w1[w_idx]:
                    num = 0
                    for v2 in self.cum_rewards:
                        num += self.cum_rewards[v2][idx][1]
                    w_idx += 1
                    window1[v].append(self.cum_rewards[v][idx][1] / num)
                if w_idx == len(w1):
                    break

        for v in self.cum_rewards:
            w_idx = 0
            for idx in range(len(self.cum_rewards[v])):
                if self.cum_rewards[v][idx][0] == w2[w_idx]:
                    num = 0
                    for v2 in self.cum_rewards:
                        num += self.cum_rewards[v2][idx][1]
                        # pdb.set_trace()
                        # print("{} ".format(self.cum_rewards[v2][idx][1]), end=" ")
                    w_idx += 1
                    # print(num)
                    window2[v].append(self.cum_rewards[v][idx][1] / num)
                if w_idx == len(w2):
                    break

        return window1, window2

    def stationarity_testing(self, user, runtime):
        print(user)
        w1, w2 = self.get_windows(runtime)
        for v in self.cum_rewards:
            kldivergence = sum(rel_entr(w1[v], w2[v]))
            print("Visualization {} KL-Divergence value {}".format(v, kldivergence))
            # for key in w1[v]:
            #     print("{:.2f} ".format(key), end=" ")
            # print()
            # for key in w2[v]:
            #     print("{:.2f} ".format(key), end=" ")
            # print()

    def plot_graph(self, user):
        for v in self.cum_rewards:
            x_axis = []
            y_axis = []
            for keys in self.cum_rewards[v]:
                x_axis.append(keys[0])
                y_axis.append(keys[1])

            plt.plot(x_axis, y_axis, label=v)
            plt.ylabel('Rewards')
            # plt.xticks([])
            plt.xlabel('time')
        plt.legend(loc='best')
        title = 'Cumulative Rewards for user: ' + user
        plt.title(title)
        plt.show()

    # def bootstrapping(self, user):

if __name__ == "__main__":
    obj = integrate()
    obj.get_files()