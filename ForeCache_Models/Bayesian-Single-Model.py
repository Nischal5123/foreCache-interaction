import environment_vizrec
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

eps = 1e-35

class Bayesian:
    def __init__(self):
        # Store action frequencies for each state
        self.freq = defaultdict(lambda: defaultdict(float))





    def train(self, user, env):
        """
        Train the model on a given user's data by learning action frequencies and rewards.
        """
        length = len(env.mem_action)
        # Train on the full user data
        for i in range(1, length):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1


    def test(self, user, env):
        """
        Test the model on a given user's data, predicting actions and evaluating accuracy.
        """
        length = len(env.mem_action)
        accuracy = []
        y_true = []
        y_pred = []
        ground_truth = []
        all_predictions = []
        insight = defaultdict(list)

        for i in range(0, length):
            try:
                # Sample based on the probability of all actions
                action_freq = [self.freq[env.mem_states[i]][action] for action in ['same', 'modify-1', 'modify-2', 'modify-3']]
                action_prob = [x / sum(action_freq) for x in action_freq]
                predicted_action = np.random.choice(['same', 'modify-1', 'modify-2', 'modify-3'], p=action_prob)

            except ZeroDivisionError:
                # If the state has not been seen, then it doesnt have action frequency --> randomly choose an action
                predicted_action = random.choice(['same', 'modify-1', 'modify-2', 'modify-3'])

            y_pred.append((predicted_action, i, user))
            y_true.append((env.mem_action[i], i, user))


            #
            ground_truth.append(env.mem_action[i])
            all_predictions.append(predicted_action)

            if predicted_action == env.mem_action[i]:
                self.freq[env.mem_states[i]][predicted_action] += 1
                insight[env.mem_action[i]].append(1)
                accuracy.append(1)
            else:
                self.freq[env.mem_states[i]][env.mem_action[i]] += 1
                insight[env.mem_action[i]].append(0)
                accuracy.append(0)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        overall_accuracy = np.mean(accuracy)
        return overall_accuracy, granular_prediction, y_pred, y_true, all_predictions, ground_truth


def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

def run_experiment(user_list, algo, task, dataset):
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Algorithm', 'StateAccuracy'])
    accuracy_all = []
    y_true_all = []
    y_pred_all = []

    # Initialize environment and algorithm
    env = environment_vizrec.environment_vizrec()


    # Leave-one-out: train on all users except the test user
    for test_user in user_list:
        obj = Bayesian()
        #print(f"Evaluating for Test User: {get_user_name(test_user)}")

        # Reset environment
        env.reset(True)

        # Training phase: train on all users except the test user
        for train_user in user_list:
            if train_user != test_user:
                env.reset(True)
                env.process_data(train_user, 0)
                obj.train(get_user_name(train_user), env)

        # Testing phase: test on the left-out test user
        env=environment_vizrec.environment_vizrec()
        env.process_data(test_user, 0)
        accuracy, granularPredictions, y_pred, y_true,all_predictions, ground_truth  = obj.test(get_user_name(test_user), env)
        accuracy_all.append(accuracy)

        # Save results
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
            'User': get_user_name(test_user),
            'Accuracy': accuracy,
            'Algorithm': algo,
            'GranularPredictions': str(granularPredictions),
            'Predictions': str(all_predictions),
            'GroundTruth': str(ground_truth),
        }])], ignore_index=True)

        #print(f"User {get_user_name(test_user)} - Accuracy: {accuracy:.2f}")

    # Save final results to CSV
    result_path = f"Experiments_Folder/VizRec/{dataset}/{task}/{algo}-Single-Model.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result_dataframe.to_csv(result_path, index=False)

    # Plot predictions and save them to a file
    #plot_predictions(y_true_all, y_pred_all, task, dataset)
    save_data_to_csv(y_true_all, y_pred_all, task, dataset)
    return accuracy_all


def format_split_accuracy(accuracy_dict):
    """
    Format accuracy per state (for specific states like Foraging, Navigation, etc.)
    """
    main_states = ['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state = []
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.nanmean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None)
    return accuracy_per_state


def save_data_to_csv(y_true_all, y_pred_all, task, dataset, algorithm='Greedy'):
    """
    Save prediction results to CSV.
    """
    data = []
    for yt, yp in zip(y_true_all, y_pred_all):
        data.append([yt[0], yt[1], yt[2], yp[0], yp[1]])

    df = pd.DataFrame(data, columns=['y_true', 'interaction_point', 'user', 'y_pred', 'pred_interaction_point'])
    directory = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
    os.makedirs(directory, exist_ok=True)
    df.to_csv(f"{directory}/{algorithm}_one_model_predictions_data.csv", index=False)


if __name__ == "__main__":
    datasets = ['birdstrikes', 'movies']
    tasks = ['p1', 'p2', 'p3', 'p4']

    overall_accuracy= []
    for dataset in datasets:
        dataset_acc = []
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = sorted(env.get_user_list(dataset, task))
            accuracy_all=run_experiment(user_list_name, 'Bayesian', task, dataset)
            print(f"Dataset: {dataset} Task: {task}, Accuracy: {np.mean(accuracy_all)}")
            dataset_acc.append(np.mean(accuracy_all))
        print(f"Dataset: {dataset}, Overall Accuracy: {np.mean(dataset_acc)}")
        overall_accuracy.append(np.mean(dataset_acc))
    print("Overall Accuracy: ", np.mean(overall_accuracy))

