import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import warnings
from collections import defaultdict
warnings.simplefilter(action='ignore', category=FutureWarning)


class environment_vizrec:
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

    def reset(self, all=False, test=False):
        """
        Reset the variables used for tracking position of the agents.

        :param all: If true, reset all variables.
        :param test: If true, set the current step to threshold value.
        :return: Current state.
        """
        if test:
            self.steps = self.threshold
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            self.mem_roi = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s

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
        for index, row in df.iterrows():
            cur_state = row['State']
            action = row['Action']
            reward = row['Reward']
            self.mem_states.append(cur_state)
            self.mem_reward.append(reward)
            self.mem_action.append(action)
            cnt_inter += 1
        #scale reward min-max
        #self.mem_reward = (self.mem_reward - np.min(self.mem_reward)) / (np.max(self.mem_reward) - np.min(self.mem_reward))


        #self.threshold = int(cnt_inter * thres)

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
class OfflineSVM:
    def __init__(self, max_iter=1000):
        """
        Initializes the Online SVM model using SGDClassifier.
        """
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-3)

    def trainOffline(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        # self.model.fit(X_train, y_train)
        self.model.fit(X_train, y_train)

    def evaluateOffline(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        predictions = []
        ground_truth = []
        # one shot prediction
        y_pred = self.model.predict(X_test)
        all_accuracies = accuracy_score(y_test, y_pred)
        insight = defaultdict(list)
        for i in range(len(y_test)):
            predictions.append(y_pred[i])
            ground_truth.append(y_test[i])
            if y_test[i] == y_pred[i]:

                insight[y_test[i]].append(1)
            else:
                insight[y_test[i]].append(0)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        return all_accuracies, granular_prediction, predictions, ground_truth




def run_experiment(user_list, dataset, task):
    valid_actions = ['same', 'modify-1', 'modify-2', 'modify-3']
    """
    Run the experiment using Leave-One-Out Cross-Validation (LOOCV).
    """
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Algorithm'])

    user_list = shuffle(user_list, random_state=42)
    user_list = list(user_list)  # Convert numpy array to Python list

    y_true_all = []
    y_pred_all = []

    for test_user_log in user_list:
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for thres in threshold:
            env = environment_vizrec()
            env.process_data(test_user_log, thres)

            length = len(env.mem_states)
            threshold_exact= int(length * thres)
            X_train, y_train = [], []

            #prepare the training data
            for i in range(threshold_exact):
                X_train.append(ast.literal_eval(env.mem_states[i]))
                y_train.append(env.mem_action[i])

            # Ensure we have more than one class to train
            if len(set(y_train)) < 1:
                # Add a default example with an additional action to avoid issues
                X_train.append([0, 0, 0])  # Example dummy feature vector
                y_train.append('same')

            if len(set(y_train)) < 2:
                # Add a default example with an additional action to avoid issues
                X_train.append([0, 0, 0])  # Example dummy feature vector
                #randomly select another action not in y_train
                y_train.append(np.random.choice([action for action in valid_actions if action not in y_train]))

            # Train the SVM on the full training set
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Initialize and train the OnlineSVM model
            model = OfflineSVM()
            model.trainOffline(X_train, y_train)

            # Test on the left-out data
            X_test, y_test = [], []
            for i in range(threshold_exact, length):
                X_test.append(ast.literal_eval(env.mem_states[i]))
                y_test.append(env.mem_action[i])

            # Evaluate the model on the test data for this user
            accuracy, granularPredictions, pred, ground_truth = model.evaluateOffline(np.array(X_test), np.array(y_test))

            # Store results
            user_name = os.path.basename(test_user_log).replace('_log.csv', '')
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Accuracy': [accuracy],
                'Algorithm': ['OfflineSVM-Individual-Model'],
                'GranularPredictions': [str(granularPredictions)],
                'Predictions': [str(pred)],
                'GroundTruth': [str(ground_truth)]

            })], ignore_index=True)



        #print(f"User {user_name} - Accuracy: {accuracy:.2f}")

    # Save results to CSV
    result_dataframe.to_csv(f"Experiments_Folder/Individual-Model/{dataset}/{task}/OfflineSVM-Individual-Model.csv", index=False)
    print(f"Dataset: {dataset} Task: {task} Algorithm: OfflineSVM-Individual-Model, Average Accuracy: {result_dataframe['Accuracy'].mean()}")



def delete_files(filename):
    """
    Delete the result files generated by the any Experiment.
    """
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2','p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            try:
                os.remove(f"Experiments_Folder/Individual-Model/{dataset}/{task}/{filename}")
            except FileNotFoundError:
              print(f"File not found: {filename}")

if __name__ == "__main__":
    # datasets = ['movies', 'birdstrikes']
    # tasks = ['p1', 'p2','p3', 'p4']
    # for dataset in datasets:
    #     for task in tasks:
    #         env = environment_vizrec()
    #         user_list = env.get_user_list(dataset, task)
    #         run_experiment(user_list, dataset, task)
    #         print(f"Done with {dataset} {task}")

    delete_files("OnlineSVM.csv")
