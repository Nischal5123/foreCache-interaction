import environment2 as environment2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import pandas as pd
import random

eps=1e-35
class Greedy:
    def __init__(self):
        """
                     Initializes the Greedy model.
                     """
        self.freq = defaultdict(lambda: defaultdict(float))
        self.reward = defaultdict(lambda: defaultdict(float))

    def GreedyDriver(self, user, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(1, threshold):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1
            self.reward[env.mem_states[i - 1]][env.mem_action[i]] += env.mem_reward[i]+eps

        # Normalizing
        for states in self.reward:
            sum = 0
            for actions in self.reward[states]:
                sum += self.reward[states][actions]
            for actions in self.reward[states]:
                self.reward[states][actions] = self.reward[states][actions] / sum
        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        y_true=[]
        y_pred=[]
        split_accuracy = defaultdict(list)
        print("threshold",threshold, "length",length-1)
        for i in range(threshold + 1, length - 1):
            denom += 1
            try:
             _max = max(self.reward[env.mem_states[i - 1]], key=self.reward[env.mem_states[i - 1]].get)
            except ValueError:
                if env.mem_states[i - 1]=='Navigation':
                    _max=random.choice(['same','change','changeout'])
                else:
                    _max = random.choice(['same', 'change'])
            y_pred.append(_max)
            y_true.append(env.mem_action[i])

            if _max == env.mem_action[i] and self.reward[env.mem_states[i - 1]][_max] > 0:
                 split_accuracy[env.mem_states[i - 1]].append(1)
                 accuracy += 1
            else:
                split_accuracy[env.mem_states[i - 1]].append(0)




        accuracy /= denom
        print("{}, {:.2f}".format(user, accuracy))
        self.freq.clear()
        self.reward.clear()
        return accuracy,split_accuracy


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
    result_dataframe = pd.DataFrame(
        columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])
    title=algo
    y_accu_all = []

    for u in user_list:
        y_accu = []
        threshold = hyperparams['threshold']
        user_name = get_user_name(u)
        for thres in threshold:
            env.process_data(u, 0)
            obj = Greedy()
            accu, state_accuracy = obj.GreedyDriver(user_name, env, thres)
            accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(accu)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [0],
                'Discount': [0],
                'Accuracy': [accu],
                'StateAccuracy': [accuracy_per_state],
                'Algorithm': [title],
                'Reward': [0]
            })], ignore_index=True)
            env.reset(True, False)
        print("User ", user_name, " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=user_name, marker='*')
        y_accu_all.append(y_accu)

    print("Greedy Model Performace: ", "Global Accuracy: ", np.mean(y_accu_all))
    # Save result DataFrame to CSV file
    result_dataframe.to_csv("Experiments_Folder/{}.csv".format(title), index=False)


if __name__ == "__main__":
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    run_experiment(user_list_2D, 'Greedy', 'sampled-hyperparameters-config.json')







