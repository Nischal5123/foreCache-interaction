import environment_vizrec
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
        accuracy = []
        denom = 0
        y_true=[]
        y_pred=[]
        split_accuracy = defaultdict(list)
        print("threshold",threshold, "length",length-1)
        for i in range(threshold + 1, length - 1):

            try:
             _max = max(self.reward[env.mem_states[i - 1]], key=self.reward[env.mem_states[i - 1]].get)
            #if state is not observed in training data then take a random action
            except ValueError:
                print('{} Not observed before'.format(env.mem_states[i-1]))
                # _max = random.choice(['same','modify-x','modify-y','modify-z','modify-x-y','modify-y-z','modify-x-z','modify-x-y-z'])
                _max = random.choice(['same', 'modify-1','modify-2','modify-3'])
                #_max = random.choice(['same', 'modify'])
            y_pred.append(_max)
            y_true.append(env.mem_action[i])

            #if state never observed before then take a random action
            if _max == env.mem_action[i]: #can also get lucky with random action
                 split_accuracy[env.mem_states[i - 1]].append(1)
                 accuracy.append(1)
            else:
                split_accuracy[env.mem_states[i - 1]].append(0)
                accuracy.append(0)

            #still learning during testing
            if _max == env.mem_action[i]:
                self.reward[env.mem_states[i - 1]][_max] += env.mem_reward[i - 1]

            denom += 1




        accuracy = np.nanmean(accuracy)
        print("{}, {:.2f}".format(user, accuracy))
        self.freq.clear()
        self.reward.clear()
        return accuracy,split_accuracy


def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.nanmean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname
def run_experiment(user_list, algo, hyperparam_file,task='p2'):
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
            #accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(accu)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [0],
                'Discount': [0],
                'Accuracy': [accu],
                'StateAccuracy': [0],
                'Algorithm': [title],
                'Reward': [0]
            })], ignore_index=True)
            env.reset(True, False)
        print("User ", user_name, " across all thresholds ", "Global Accuracy: ", np.nanmean(y_accu))

        plt.plot(threshold, y_accu, label=user_name, marker='*')
        y_accu_all.append(y_accu)

    print("Greedy Model Performace: ", "Global Accuracy: ", np.nanmean(y_accu_all))
    # Save result DataFrame to CSV file
    result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}.csv".format(task,title), index=False)


if __name__ == "__main__":
    task = 'p4'
    env = environment_vizrec.environment_vizrec()
    user_list_2D = env.user_list_2D
    print(user_list_2D)
    run_experiment(user_list_2D, 'Greedy', 'sampled-hyperparameters-config.json',task)







