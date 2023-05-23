import environment2 as environment2
import numpy as np
from collections import defaultdict
import pdb
import misc
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random

class Naive:
    def __init__(self):
        """
               Initializes the Naive object.
               """
        self.bestaction = defaultdict(
            lambda: defaultdict(str)
        )  # Initializes a dictionary with default values
        self.reward = defaultdict(
            lambda: defaultdict(float)
        )  # Initializes a dictionary with default values
        self.states = [
            "Sensemaking",
            "Foraging",
            "Navigation",
        ]  # Defines the possible states of the environment
        self.actions = [
            "change",
            "same",
            "changeout",
        ]  # Defines the possible actions of the agent
        for state in self.states:
            self.bestaction[
                state
            ] = self.take_random_action(state, "")
            for action in self.actions:
                self.reward[state][action] = 0
    def take_random_action(self, state, action):
        """
        Selects a random action different from the current one.

        Args:
        - state (str): the current state of the environment.
        - action (str): the current action taken by the agent.

        Returns:
        - next_action (str): a randomly chosen action different from the current one.
        """
        if state == "Navigation":
            action_space = ["same", "change", "changeout"]
            action_space = [f for f in action_space if f != action]
            next_action = random.choice(action_space)
        else:
            action_space = ["same", "change"]
            action_space = [f for f in action_space if f != action]
            next_action = random.choice(action_space)
        return next_action
    def NaiveProbabilistic(self, user, env, thres):
        """
               Implements the Win-Stay Lose-Switch algorithm for a given user and environment.

               Args:
               - user (list): a list containing the data of a given user.
               - env (environment2): an environment object.
               - thres (float): the threshold value.

               Returns:
               - accuracy (float): the accuracy of the algorithm.
               """
        length = len(env.mem_action)
        threshold = int(length * thres)

        accuracy = 0
        denom = 0

        result = []
        accuracy = []
        split_accuracy = defaultdict(list)
        for i in range(threshold + 1, length - 1):
            cur_action = self.bestaction[env.mem_states[i]]
            result.append(env.mem_states[i])
            result.append(cur_action)
            if self.bestaction[env.mem_states[i]] == env.mem_action[i]:
                accuracy.append(1)
                split_accuracy[env.mem_states[i]].append(1)
            else:
                split_accuracy[env.mem_states[i]].append(0)
                accuracy.append(0)
            denom += 1

        obj = misc.misc([])
        print("{}, {:.2f}, {}".format(obj.get_user_name(user), np.mean(accuracy), result))
        self.bestaction.clear()
        self.reward.clear()
        return np.mean(accuracy), split_accuracy


def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(0)
    return accuracy_per_state

def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname


if __name__ == "__main__":

    result_dataframe = pd.DataFrame(
        columns=['User', 'Accuracy','Threshold', 'LearningRate', 'Discount','Algorithm','StateAccuracy'])

    dataframe_users = []
    dataframe_threshold = []
    dataframe_learningrate = []
    dataframe_accuracy =[]
    dataframe_discount = []
    dataframe_accuracy_per_state = []
    dataframe_algorithm=[]



    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    total = 0
    threshold = [0.1]
    obj2 = misc.misc([])
    y_accu_all=[]

    for u in user_list_2D:
        y_accu = []
        threshold=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]




        for thres in threshold:
            env.process_data(u, 0)
            obj = Naive()
            accu,state_accuracy = obj.NaiveProbabilistic(u, env, thres)
            accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(accu)
            dataframe_users.append(get_user_name(u))
            dataframe_threshold.append(thres)
            dataframe_learningrate.append(0)
            dataframe_discount.append(0)
            dataframe_accuracy.append(accu)
            dataframe_accuracy_per_state.append(accuracy_per_state)
            dataframe_algorithm.append("Naive")
            env.reset(True, False)
        print("User ", get_user_name(u), " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=obj2.get_user_name(u), marker='*')
        y_accu_all.append(np.mean(y_accu))
   # plt.yticks(np.arange(0.0, 1.0, 0.1))
    print("Naive Model PERFORMACE: ", "Global Accuracy: ", np.mean(y_accu_all))
    title = "Naive"
    result_dataframe['User'] = dataframe_users
    result_dataframe['Threshold'] = dataframe_threshold
    result_dataframe['LearningRate'] = dataframe_learningrate
    result_dataframe['Discount'] = dataframe_discount
    result_dataframe['Accuracy'] = dataframe_accuracy
    result_dataframe['Algorithm']=dataframe_algorithm
    result_dataframe['StateAccuracy']=dataframe_accuracy_per_state
    result_dataframe.to_csv("Experiments_Folder\\" + title + ".csv", index=False)


   #
   # def NaiveProbabilistic(self, user, env, thres):
   #      """
   #             Implements the Win-Stay Lose-Switch algorithm for a given user and environment.
   #
   #             Args:
   #             - user (list): a list containing the data of a given user.
   #             - env (environment2): an environment object.
   #             - thres (float): the threshold value.
   #
   #             Returns:
   #             - accuracy (float): the accuracy of the algorithm.
   #             """
   #      length = len(env.mem_action)
   #      threshold = int(length * thres)
   #      multi_result = []
   #      multi_accuracy = []
   #      multi_split_accuracy = defaultdict(list)
   #      for repeat in range(5):
   #          accuracy = 0
   #          denom = 0
   #
   #          result = []
   #          accuracy = []
   #          split_accuracy = defaultdict(list)
   #
   #          for i in range(threshold + 1, length - 1):
   #              cur_action = self.bestaction[env.mem_states[i]]
   #              result.append(env.mem_states[i])
   #              result.append(cur_action)
   #              if self.bestaction[env.mem_states[i]] == env.mem_action[i]:
   #                  accuracy.append(1)
   #                  split_accuracy[env.mem_states[i]].append(1)
   #              else:
   #                  split_accuracy[env.mem_states[i]].append(0)
   #                  accuracy.append(0)
   #              denom += 1
   #
   #          obj = misc.misc([])
   #          print("{}, {:.2f}, {}".format(obj.get_user_name(user), np.mean(accuracy), result))
   #          self.bestaction.clear()
   #          self.reward.clear()
   #          multi_accuracy.append(np.mean(accuracy))
   #          for state in self.states:
   #              multi_split_accuracy[state].append(self.format_split_accuracy(split_accuracy,state))
   #      return np.mean(multi_accuracy), multi_split_accuracy
   #
   #  def format_split_accuracy(self,accuracy_dict, state=None):
   #      if state == None:
   #          main_states = ['Foraging', 'Navigation', 'Sensemaking']
   #          accuracy_per_state = []
   #          for state in main_states:
   #              if accuracy_dict[state]:
   #                  accuracy_per_state.append(np.mean(accuracy_dict[state]))
   #              else:
   #                  accuracy_per_state.append(0)
   #      else:
   #          accuracy_per_state = np.mean(accuracy_dict[state])
   #
   #      return accuracy_per_state
