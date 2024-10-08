import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Momentum:
    """
    Momentum class: Tracks the best action taken in each state.
    """
    def __init__(self):
        """
        Initializes the Momentum object.
        """
        self.bestaction = defaultdict()  # Initializes a dictionary for the best action per state
        self.actions = ['same', 'modify-1', 'modify-2', 'modify-3']  # Available actions

    def take_random_action(self, state):
        """
        Selects a random action.

        Args:
        - state (str): the current state of the environment.

        Returns:
        - next_action (str): a randomly chosen action.
        """
        return random.choice(self.actions)

    def MomentumDriver(self, user, env):
        """
        Executes the Momentum strategy for a given user and environment.

        Args:
        - user (str): The user being tested.
        - env (object): The environment object.

        Returns:
        - accuracy (float): Accuracy of predicted actions compared to the actual actions.
        """
        length = len(env.mem_action)
        correct_predictions = 0
        ground_truth = []
        all_predictions = []
        insight = defaultdict(list)

        threshold = int(length * 0.8)
        for i in range(threshold):
            self.bestaction[env.mem_states[i]] = env.mem_action[i]




        for i in range(threshold, length):
            current_state = env.mem_states[i]

            # If a best action exists for the state, use it, otherwise take a random action
            try:
                action_to_take = self.bestaction[current_state]
            except KeyError:
                action_to_take = self.take_random_action(current_state)

            ground_truth.append(env.mem_action[i])
            all_predictions.append(action_to_take)
            # Check if the action matches
            if action_to_take == env.mem_action[i]:
                correct_predictions += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)

            # Update the best action based on ground truth for the state
            self.bestaction[current_state] = env.mem_action[i]

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        # Calculate accuracy
        accuracy = correct_predictions / length if length > 0 else 0
        return accuracy,  granular_prediction, all_predictions, ground_truth

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname


def get_last_20_actions_accuracy(ground_truth, all_predictions):
    """
    Calculates the accuracy of the last 20 actions.

    Args:
    - ground_truth (list): The list of actual actions.
    - all_predictions (list): The list of predicted actions.

    Returns:
    - accuracy (float): The accuracy of the last 20 actions.
    """
    twentyPercent = int(len(ground_truth) * 0.2)
    ground_truth_last_20 = ground_truth[-twentyPercent:]
    all_predictions_last_20 = all_predictions[-twentyPercent:]
    correct_predictions = sum(
        [1 for i in range(twentyPercent) if ground_truth_last_20[i] == all_predictions_last_20[i]])
    return correct_predictions / twentyPercent if twentyPercent > 0 else 0

if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []
    last_twenty_accuracy = []
    for dataset in datasets:
        dataset_acc = []
        for task in tasks:
            # Initialize environment and get user list
            env = environment_vizrec.environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            result_dataframe = pd.DataFrame(
                columns=['User', 'Accuracy', 'Algorithm', 'GranularPredictions', 'Predictions', 'GroundTruth']
            )

            # Perform leave-one-out cross-validation
            task_accuracy = []
            for test_user in user_list_name:
                #print(f"Evaluating for Test User: {get_user_name(test_user)}")

                # Initialize environment for the current user
                env = environment_vizrec.environment_vizrec()
                obj = Momentum()

                # Process test user data
                env.process_data(test_user, 0)  # Load test user data
                accuracy, granularPredictions,all_predictions, ground_truth = obj.MomentumDriver(test_user, env)  # Evaluate on the test user
                #last_twenty_accuracy.append(get_last_20_actions_accuracy(ground_truth, all_predictions))

                task_accuracy.append(accuracy)

                dataset_acc.append(accuracy)

                # Save results
                result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                    'User': get_user_name(test_user),
                    'Accuracy': accuracy,
                    'Algorithm': 'Momentum',
                    'GranularPredictions': str(granularPredictions),
                    'Predictions': str(all_predictions),
                    'GroundTruth': str(ground_truth),
                }])], ignore_index=True)

                #print(f"User {get_user_name(test_user)} - Accuracy: {accuracy:.2f}")


            # Save results to CSV
            result_path = f"Experiments_Folder/VizRec/{dataset}/{task}/Momentum-Single-Model.csv"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_dataframe.to_csv(result_path, index=False)
            print(f"Dataset: {dataset} Task: {task}, Accuracy: {np.mean(task_accuracy)}")
        print(f"Dataset: {dataset}, Overall Accuracy: {np.mean(dataset_acc)}")
        overall_accuracy.append(np.mean(dataset_acc))
    print(f"Overall Accuracy: {np.mean(overall_accuracy)}")
    #print(f"Last 20 Actions Accuracy: {np.mean(last_twenty_accuracy)}")




