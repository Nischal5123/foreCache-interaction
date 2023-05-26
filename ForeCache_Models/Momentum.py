import environment2 as environment2
import numpy as np
from collections import defaultdict
import misc
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random
import json

class Momentum:
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
        Selects a random action based on specified probabilities.

        Args:
        - state (str): the current state of the environment.
        - action (str): the current action taken by the agent.

        Returns:
        - next_action (str): a randomly chosen action based on the specified probabilities.
        """
        if state == "Navigation":
            action_space = ["same", "change", "changeout"]
            probabilities = [0.9 if f == action else 0.05 for f in action_space]
        else:
            action_space = ["same", "change"]
            probabilities = [0.9 if f == action else 0.1 for f in action_space]

        next_action = random.choices(action_space, probabilities)[0]
        return next_action
    def MomentumProbabilistic(self, user, env, thres):
        """
               Implements the Momentum algorithm for a given user and environment.

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

        #initialize last best action from training data
        for j in range(0,threshold):
            self.bestaction[env.mem_states[j]] = env.mem_action[j]

        #testing data
        for i in range(threshold + 1, length - 1):
            cur_action=self.bestaction[env.mem_states[i]]
            if cur_action == env.mem_action[i]:
                accuracy.append(1)
                split_accuracy[env.mem_states[i]].append(1)
            else:
                split_accuracy[env.mem_states[i]].append(0)
                accuracy.append(0)
            denom += 1
            self.bestaction[env.mem_states[i]] = self.take_random_action(env.mem_states[i], self.bestaction[env.mem_states[i]])
            result.append(env.mem_states[i])
            result.append(cur_action)

        obj = misc.misc([])
        print("{}, {:.2f}, {}".format(user, np.mean(accuracy), result))
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
    string = url.split('/')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname


def run_experiment(user_list, algo, hyperparam_file):
    # Load hyperparameters from JSON file
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy','Threshold', 'LearningRate', 'Discount','Algorithm','StateAccuracy'])
    y_accu_all=[]
    title=algo
    for u in user_list:
        y_accu = []
        threshold=hyperparams['threshold']
        user_name = get_user_name(u)
        for thres in threshold:
            env.process_data(u, 0)
            obj = Momentum()
            test_accuracy,state_accuracy = obj.MomentumProbabilistic(user_name, env, thres)
            accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(test_accuracy)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [0],
                'Discount': [0],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [accuracy_per_state],
                'Algorithm': [title],
                'Reward': [0]
            })], ignore_index=True)
            env.reset(True, False)
        print("User ", get_user_name(u), " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=user_name, marker='*')
        y_accu_all.append(np.mean(y_accu))

    print("Momentum Model Performace: ", "Global Accuracy: ", np.mean(y_accu_all))
    # Save result DataFrame to CSV file
    result_dataframe.to_csv("Experiments_Folder/{}.csv".format(title), index=False)


if __name__ == "__main__":
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    run_experiment(user_list_2D, 'Momentum', 'sampled-hyperparameters-config.json')
