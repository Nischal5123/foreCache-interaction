import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class WSLS:
    """
    WSLS class: Tracks the best action taken in each state.
    """
    def __init__(self):
        """
        Initializes the WSLS object.
        """
        self.bestaction = defaultdict(lambda: None)  # Initializes a dictionary for the best action per state
        self.actions = ['same', 'modify-1', 'modify-2', 'modify-3']  # Available actions
        self.reward = defaultdict(lambda: defaultdict(float))

    def take_random_action(self, state, action):
        """
        Selects a random action different from the current one.

        Args:
        - state (str): the current state of the environment.
        - action (str): the current action taken by the agent.

        Returns:
        - next_action (str): a randomly chosen action different from the current one.
        """

        action_space = [f for f in self.actions if f != action]
        next_action = random.choice(action_space)
        return next_action

    def WSLSDriver(self, user, env):
        """
        Executes the WSLS strategy for a given user and environment.

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
        result=[]



        for i in range(length):
            if self.bestaction[env.mem_states[i]] is None:
                action = random.choice(self.actions)
                self.bestaction[env.mem_states[i]] = action
                result.append("Random")
            cur_action = self.bestaction[env.mem_states[i]]
            result.append(env.mem_states[i])
            result.append(cur_action)

            all_predictions.append(cur_action)
            ground_truth.append(env.mem_action[i])

            if env.mem_reward[i] > (self.reward[env.mem_states[i]][cur_action]):
                result.append("Win")
                # current action is best if win
                action = cur_action

                self.reward[env.mem_states[i]][action] = env.mem_reward[i]
                self.bestaction[env.mem_states[i]] = action
            else:
                # chnage from other actions in loose
                self.bestaction[env.mem_states[i]] = self.take_random_action(
                    env.mem_states[i], cur_action)
                result.append("Loose")
            # after deciding on statying with current action or switching calculate accuracy

        # performance book-keeping
            if cur_action == env.mem_action[i]:
                correct_predictions += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)
            ground_truth.append(env.mem_action[i])
            all_predictions.append(cur_action)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        # Calculate accuracy
        accuracy = correct_predictions / length if length > 0 else 0
        return accuracy, granular_prediction, all_predictions, ground_truth



          # Calculate accuracy
        accuracy = correct_predictions / length if length > 0 else 0
        return accuracy,  all_predictions, ground_truth

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
                obj = WSLS()

                # Process test user data
                env.process_data(test_user, 0)  # Load test user data
                accuracy, granularPredictions, all_predictions, ground_truth = obj.WSLSDriver(test_user, env)  # Evaluate on the test user
                task_accuracy.append(accuracy)

                dataset_acc.append(accuracy)

                # Save results
                # Save results
                result_dataframe = pd.concat([result_dataframe, pd.DataFrame([{
                    'User': get_user_name(test_user),
                    'Accuracy': accuracy,
                    'Algorithm': 'WSLS',
                    'GranularPredictions': str(granularPredictions),
                    'Predictions': str(all_predictions),
                    'GroundTruth': str(ground_truth),
                }])], ignore_index=True)

                #print(f"User {get_user_name(test_user)} - Accuracy: {accuracy:.2f}")


            # Save results to CSV
            result_path = f"Experiments_Folder/VizRec/{dataset}/{task}/WSLS-Single-Model.csv"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_dataframe.to_csv(result_path, index=False)
            print(f"Dataset: {dataset} Task: {task}, Accuracy: {np.mean(task_accuracy)}")
        print(f"Dataset: {dataset}, Overall Accuracy: {np.mean(dataset_acc)}")
        overall_accuracy.append(np.mean(dataset_acc))
    print(f"Overall Accuracy: {np.mean(overall_accuracy)}")


