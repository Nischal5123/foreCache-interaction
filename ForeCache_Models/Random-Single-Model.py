import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Random:
    """
    Random class: Tracks the best action taken in each state.
    """
    def __init__(self):
        """
        Initializes the Random object.
        """
        self.bestaction = defaultdict(lambda: None)  # Initializes a dictionary for the best action per state
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

    def RandomDriver(self, user, env):
        """
        Executes the Random strategy for a given user and environment.

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



        for i in range(length):
            current_state = env.mem_states[i]

            action_to_take = self.take_random_action(current_state)

            # Check if the action matches
            if action_to_take == env.mem_action[i]:
                correct_predictions += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)
            ground_truth.append(env.mem_action[i])
            all_predictions.append(action_to_take)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        # Calculate accuracy
        accuracy = correct_predictions / length if length > 0 else 0
        return accuracy, granular_prediction, all_predictions, ground_truth

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname



if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []
    for dataset in datasets:
        dataset_acc = []
        for task in tasks:
            # Initialize environment and get user list
            env = environment_vizrec.environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            result_dataframe = pd.DataFrame(
                columns=['User', 'Accuracy', 'Algorithm']
            )

            # Perform leave-one-out cross-validation
            task_accuracy = []
            for test_user in user_list_name:
                #print(f"Evaluating for Test User: {get_user_name(test_user)}")

                # Initialize environment for the current user
                env = environment_vizrec.environment_vizrec()
                obj = Random()

                # Process test user data
                env.process_data(test_user, 0)  # Load test user data
                accuracy, granularPredictions, all_predictions, ground_truth = obj.RandomDriver(test_user, env)  # Evaluate on the test user
                task_accuracy.append(accuracy)

                dataset_acc.append(accuracy)

                # Save results
                result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                    'User': get_user_name(test_user),
                    'Accuracy': accuracy,
                    'Algorithm': 'Random',
                    'GranularPredictions': str(granularPredictions),
                    'Predictions': str(all_predictions),
                    'GroundTruth': str(ground_truth),
                }])], ignore_index=True)

                #print(f"User {get_user_name(test_user)} - Accuracy: {accuracy:.2f}")


            # Save results to CSV
            result_path = f"Experiments_Folder/VizRec/{dataset}/{task}/Random-Single-Model.csv"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_dataframe.to_csv(result_path, index=False)
            print(f"Dataset: {dataset} Task: {task}, Accuracy: {np.mean(task_accuracy)}")
        print(f"Dataset: {dataset}, Overall Accuracy: {np.mean(dataset_acc)}")
        overall_accuracy.append(np.mean(dataset_acc))
    print(f"Overall Accuracy: {np.mean(overall_accuracy)}")


