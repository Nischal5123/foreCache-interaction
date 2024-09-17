import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import sklearn.exceptions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class environment_vizrec:
    # Assuming the rest of the environment class remains the same
    def __init__(self):
        """
        Constructor for the environment class.

        Initializes required variables and stores data in memory.
        """
        self.user_list_movies_p1 = np.sort(glob.glob("data/zheng/processed_interactions_p1/*"))
        self.user_location_movies_p1 = "data/zheng/processed_interactions_p1/"
        self.user_list_movies_p2 = np.sort(glob.glob("data/zheng/processed_interactions_p2/*"))
        self.user_location_movies_p2 = "data/zheng/processed_interactions_p2/"
        self.user_list_movies_p3 = np.sort(glob.glob("data/zheng/processed_interactions_p3/*"))
        self.user_location_movies_p3 = "data/zheng/processed_interactions_p3/"
        self.user_list_movies_p4 = np.sort(glob.glob("data/zheng/processed_interactions_p4/*"))
        self.user_location_movies_p4 = "data/zheng/processed_interactions_p4/"
        self.user_list_birdstrikes_p1 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p1/*"))
        self.user_location_birdstrikes_p1 = "data/zheng/birdstrikes_processed_interactions_p1/"
        self.user_list_birdstrikes_p2 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p2/*"))
        self.user_location_birdstrikes_p2 = "data/zheng/birdstrikes_processed_interactions_p2/"
        self.user_list_birdstrikes_p3 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p3/*"))
        self.user_location_birdstrikes_p3 = "data/zheng/birdstrikes_processed_interactions_p3/"
        self.user_list_birdstrikes_p4 = np.sort(glob.glob("data/zheng/birdstrikes_processed_interactions_p4/*"))
        self.user_location_birdstrikes_p4 = "data/zheng/birdstrikes_processed_interactions_p4/"


        # This variable will be used to track the current position of the user agent.
        self.steps = 0
        self.done = False  # Done exploring the current subtask

        # # List of valid actions and states
        # self.valid_actions = ['same','modify-x','modify-y','modify-z','modify-x-y','modify-y-z','modify-x-z','modify-x-y-z']
        #self.valid_actions = ['same', 'modify']
        self.valid_actions = ['same', 'modify-1', 'modify-2','modify-3']
        # self.valid_states = ['Task_Panel','Related_View','Top_Panel','Data_View']


        # Storing the data into main memory. Focus is now only on action and states for a fixed user's particular subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.mem_roi = []
        self.threshold = 0
        self.prev_state = None
    def process_data(self, filename, thres):
        """
        Processes the data from the given file and stores it in the object memory.

        Inputs:
        - filename (str): path of the data file.
        - thres (float): threshold value to determine the end of the episode.

        Returns: None.
        """
        df = pd.read_csv(filename)
        cnt_inter = 0
        total_len = len(df)
        threshold_idx = int(total_len * thres)

        for index, row in df.iterrows():
            cur_state = row['State']
            action = row['Action']
            reward = row['Reward']
            self.mem_states.append(cur_state)
            self.mem_reward.append(reward)
            self.mem_action.append(action)
            cnt_inter += 1

        # Training portion ends at the threshold
        self.training_data_states = self.mem_states[:threshold_idx]
        self.training_data_actions = self.mem_action[:threshold_idx]

        # Testing portion starts after the threshold
        self.testing_data_states = self.mem_states[threshold_idx:]
        self.testing_data_actions = self.mem_action[threshold_idx:]

    def get_user_list(self,dataset,task):
        if dataset == 'movies':
            if task == 'p1':
                return self.user_list_movies_p1
            elif task == 'p2':
                return self.user_list_movies_p2
            elif task == 'p3':
                return self.user_list_movies_p3
            elif task == 'p4':
                return self.user_list_movies_p4
        elif dataset == 'birdstrikes':
            if task == 'p1':
                return self.user_list_birdstrikes_p1
            elif task == 'p2':
                return self.user_list_birdstrikes_p2
            elif task == 'p3':
                return self.user_list_birdstrikes_p3
            elif task == 'p4':
                return self.user_list_birdstrikes_p4

    def get_user_location(self,dataset,task):
        if dataset == 'movies':
            if task == 'p1':
                return self.user_location_movies_p1
            elif task == 'p2':
                return self.user_location_movies_p2
            elif task == 'p3':
                return self.user_location_movies_p3
            elif task == 'p4':
                return self.user_location_movies_p4
        elif dataset == 'birdstrikes':
            if task == 'p1':
                return self.user_location_birdstrikes_p1
            elif task == 'p2':
                return self.user_location_birdstrikes_p2
            elif task == 'p3':
                return self.user_location_birdstrikes_p3
            elif task == 'p4':
                return self.user_location_birdstrikes_p4
class OnlineSVM:
    def __init__(self, max_iter=10000):
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-3)
        self.is_fitted = False  # Flag to check if the model has been fitted

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        Ensure the model has been fitted at least once.
        """
        if len(X_train) > 0:  # Ensure there is training data
            self.model.partial_fit(X_train, y_train, classes=['modify-1', 'modify-2', 'modify-3', 'same'])
            self.is_fitted = True  # Set flag to indicate the model is fitted

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        Only evaluate if the model has been fitted.
        """
        if not self.is_fitted:  # Check if the model has been fitted
            raise sklearn.exceptions.NotFittedError("This OnlineSVM instance is not fitted yet.")

        all_accuracies = []
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            accuracy = accuracy_score([y_test[i]], y_pred)
            self.model.partial_fit([X_test[i]], [y_test[i]])  # Continue online training
            all_accuracies.append(accuracy)

        return np.mean(all_accuracies), y_pred



def run_experiment(user_list, dataset, task):
    """
    Train on progressively larger portions of each user's data and test on the remaining part.
    """
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Threshold'])

    user_list = shuffle(user_list, random_state=42)
    user_list = list(user_list)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for test_user_log in user_list:
        user_name = os.path.basename(test_user_log).replace('_log.csv', '')

        # Iterate over each threshold
        for thres in thresholds:
            env = environment_vizrec()
            env.process_data(test_user_log, thres)

            # Prepare training data (up to threshold %)
            X_train = [ast.literal_eval(state) for state in env.training_data_states]
            y_train = env.training_data_actions

            # Initialize and train the OnlineSVM model
            model = OnlineSVM()
            model.train(X_train, y_train)

            # Prepare testing data (remaining data after threshold)
            X_test = [ast.literal_eval(state) for state in env.testing_data_states]
            y_test = env.testing_data_actions

            # Evaluate the model on the test data for this user
            try:
                accuracy, _ = model.evaluate(X_test, y_test)
            except sklearn.exceptions.NotFittedError:
                accuracy = None

            # Store results
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Accuracy': [accuracy],
                'Threshold': [thres]
            })], ignore_index=True)

            #print(f"User {user_name} - Threshold: {thres} - Accuracy: {accuracy}")

    # Save results to CSV
    os.makedirs(f"Experiments_Folder/VizRec/{dataset}/{task}", exist_ok=True)
    result_dataframe.to_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/SVM_Threshold.csv", index=False)
    print(f"Experiment results saved for {dataset} {task}.")

    # Calculate and print task-level accuracy
    task_accuracy = np.nanmean(result_dataframe['Accuracy'].values)
    print(f"Task {task} - Average Accuracy: {task_accuracy}")

    return task_accuracy


if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracies = []

    for dataset in datasets:
        dataset_accuracies = []

        for task in tasks:
            env = environment_vizrec()
            user_list = env.get_user_list(dataset, task)
            task_accuracy = run_experiment(user_list, dataset, task)
            dataset_accuracies.append(task_accuracy)
            print(f"Done with {dataset} {task}")

        # Dataset-level accuracy
        dataset_average_accuracy = np.nanmean(dataset_accuracies)
        print(f"Dataset {dataset} - Average Accuracy: {dataset_average_accuracy}")

        overall_accuracies.extend(dataset_accuracies)

    # Overall accuracy across all datasets and tasks
    overall_average_accuracy = np.nanmean(overall_accuracies)
    print(f"Overall Average Accuracy: {overall_average_accuracy}")