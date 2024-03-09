import environment_vizrec
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import random
import json

class Momentum:
    def __init__(self):
        """
               Initializes the Momentum object.
               """
        self.bestaction = defaultdict(
            lambda: defaultdict(str)
        )  # Initializes a dictionary with default values
        self.reward = defaultdict(
            lambda: defaultdict(float)
        )  # Initializes a dictionary with default values
        self.states = [
        ]  # Defines the possible states of the environment
        #self.actions = ['same','modify-x','modify-y','modify-z','modify-x-y','modify-y-z','modify-x-z','modify-x-y-z']  # Defines the possible actions of the agent
        self.valid_actions = ['same', 'modify-1', 'modify-2','modify-3']
        #self.valid_actions = ['same', 'modify']
        for state in self.states:
            self.bestaction[
                state
            ] = self.take_random_action(state, "")
    def take_last_action(self, state, action):
        """
        Selects a random action different from the current one.

        Args:
        - state (str): the current state of the environment.
        - action (str): the current action taken by the agent.

        Returns:
        - next_action (str): a randomly chosen action different from the current one.
        """
        #action_space = ['same','modify-x','modify-y','modify-z','modify-x-y','modify-y-z','modify-x-z','modify-x-y-z']
        action_space = [f for f in self.valid_actions if f != action]
        next_action = random.choice(action_space)
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
        denom = 0

        result = []
        accuracy = []
        split_accuracy = defaultdict(list)

        print("threshold", threshold, "length", length - 1)
        #testing data: no training required
        last_action= random.choice(self.valid_actions)
        for i in range(threshold + 1, length - 1):
            #always take last action irrespective of the state
            cur_action=last_action
            if cur_action == env.mem_action[i]:
                accuracy.append(1)
                split_accuracy[env.mem_states[i]].append(1)
            else:

                split_accuracy[env.mem_states[i]].append(0)
                accuracy.append(0)
            denom += 1
            last_action = env.mem_action[i]
            result.append(env.mem_states[i])
            result.append(cur_action)

        print("{}, {:.2f}".format(user, np.mean(accuracy)))
        self.bestaction.clear()
        return np.mean(accuracy), split_accuracy


def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname


def run_experiment(user_list, algo, hyperparam_file,task, dataset):
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
            test_accs = []
            for i in range(5):
                env = environment_vizrec.environment_vizrec()
                env.process_data(u, 0)
                obj = Momentum()
                test_accuracy, state_accuracy = obj.MomentumProbabilistic(user_name, env, thres)
                test_accs.append(test_accuracy)
            test_accuracy = np.mean(test_accs)
            #accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(test_accuracy)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [None],
                'Discount': [None],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [0],
                'Algorithm': [title],
                'Reward': [None]
            })], ignore_index=True)
            env.reset(True, False)
        print("User ", user_name, " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=user_name, marker='*')
        y_accu_all.append(np.mean(y_accu))

    print("Momentum Model Performace: ", "Global Accuracy: ", np.mean(y_accu_all))
    # Save result DataFrame to CSV file
    result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset,task,title), index=False)



if __name__ == "__main__":
    datasets = ['movies','birdstrikes']
    tasks =['p1', 'p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            run_experiment(user_list_name, 'Momentum', 'sampled-hyperparameters-config.json', task, dataset)
            print(f"Done with {dataset} {task}")
