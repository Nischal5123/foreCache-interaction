import environment_vizrec
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import pandas as pd
import random
import os

eps = 1e-35

class Greedy:
    def __init__(self):
        self.freq = defaultdict(lambda: defaultdict(float))
        self.reward = defaultdict(lambda: defaultdict(float))

    def GreedyDriver(self, user, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(1, threshold):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1
            self.reward[env.mem_states[i - 1]][env.mem_action[i]] += env.mem_reward[i] + eps

        for states in self.reward:
            sum = 0
            for actions in self.reward[states]:
                sum += self.reward[states][actions]
            for actions in self.reward[states]:
                self.reward[states][actions] = self.reward[states][actions] / sum

        accuracy = []
        denom = 0
        y_true = []
        y_pred = []
        split_accuracy = defaultdict(list)
        print("threshold", threshold, "length", length - 1)
        for i in range(threshold, length):
            try:
                _max = max(self.reward[env.mem_states[i - 1]], key=self.reward[env.mem_states[i - 1]].get)
            except ValueError:
                print('{} Not observed before'.format(env.mem_states[i - 1]))
                _max = random.choice(['same', 'modify-1', 'modify-2', 'modify-3'])
            y_pred.append((_max, i, user, thres))
            y_true.append((env.mem_action[i], i, user, thres))

            if _max == env.mem_action[i]:
                split_accuracy[env.mem_states[i - 1]].append(1)
                accuracy.append(1)
            else:
                split_accuracy[env.mem_states[i - 1]].append(0)
                accuracy.append(0)

            if _max == env.mem_action[i]:
                self.reward[env.mem_states[i - 1]][_max] += env.mem_reward[i - 1]

            denom += 1

        accuracy = np.sum(accuracy) / denom
        print("{}, {:.2f}".format(user, accuracy))
        self.freq.clear()
        self.reward.clear()
        return accuracy, split_accuracy, y_pred, y_true

def format_split_accuracy(accuracy_dict):
    main_states = ['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state = []
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.nanmean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None)
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

def run_experiment(user_list, algo, hyperparam_file, task, dataset):
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])
    title = algo
    y_accu_all = []

    y_true_all = []
    y_pred_all = []

    for u in user_list:
        y_accu = []
        threshold = hyperparams['threshold']
        user_name = get_user_name(u)
        for thres in threshold:
            env.process_data(u, 0)
            obj = Greedy()
            accu, state_accuracy, y_pred, y_true = obj.GreedyDriver(user_name, env, thres)
            y_accu.append(accu)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
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
        y_accu_all.append(y_accu)

    print("Greedy Model Performance: ", "Global Accuracy: ", np.mean(y_accu_all))
    result_dataframe.to_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/{title}.csv", index=False)

    #plot_predictions(y_true_all, y_pred_all, task, dataset)
    save_data_to_csv(y_true_all, y_pred_all, task, dataset)

def plot_predictions(y_true_all, y_pred_all, task, dataset, algorithm='Greedy'):
    colors = {'same': 'blue', 'modify-1': 'green', 'modify-2': 'red', 'modify-3': 'purple'}
    matches = defaultdict(lambda: defaultdict(int))
    users = list(set(yt[2] for yt in y_true_all))
    users.sort()
    for yt, yp in zip(y_true_all, y_pred_all):
        if yt[0] == yp[0]:  # If prediction matches true value
            matches[(yt[1], yt[2])][yt[0]] += 1

    # Prepare data for heatmap
    heatmap_data = np.zeros((len(users), max(yt[1] for yt in y_true_all) + 1))

    for (interaction_point, user), pred_dict in matches.items():
        for count in pred_dict.values():
            user_idx = users.index(user)  # Map user to y-axis
            heatmap_data[user_idx, interaction_point] = count

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Number of Matches')

    plt.yticks(range(len(users)), users)
    plt.xlabel('Interaction Point')
    plt.ylabel('User')
    plt.title(f'Prediction Matches Heatmap for Task {task} in Dataset {dataset}')

    directory = f"Experiments_Folder/VizRec/{dataset}/{task}/plots"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/{algorithm}_all_users_y_pred_vs_y_true_heatmap.png")
    plt.close()

def save_data_to_csv(y_true_all, y_pred_all, task, dataset, algorithm='Greedy'):
    data = []
    for yt, yp in zip(y_true_all, y_pred_all):
        data.append([yt[0], yt[1], yt[2], yp[0], yp[1], yt[3]])

    df = pd.DataFrame(data, columns=['y_true', 'interaction_point', 'user', 'y_pred', 'pred_interaction_point','threshold'])
    directory = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
    os.makedirs(directory, exist_ok=True)
    df.to_csv(f"{directory}/{algorithm}_predictions_data.csv", index=False)

if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = sorted(env.get_user_list(dataset, task))
            run_experiment(user_list_name, 'Greedy', 'sampled-hyperparameters-config.json', task, dataset)
            print(f"Done with {dataset} {task}")
