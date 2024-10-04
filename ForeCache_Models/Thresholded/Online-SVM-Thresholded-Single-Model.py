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
        self.user_list_movies_p1 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p1/*"))
        self.user_location_movies_p1 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p1/"
        self.user_list_movies_p2 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p2/*"))
        self.user_location_movies_p2 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p2/"
        self.user_list_movies_p3 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p3/*"))
        self.user_location_movies_p3 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p3/"
        self.user_list_movies_p4 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p4/*"))
        self.user_location_movies_p4 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/processed_interactions_p4/"
        self.user_list_birdstrikes_p1 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p1/*"))
        self.user_location_birdstrikes_p1 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p1/"
        self.user_list_birdstrikes_p2 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p2/*"))
        self.user_location_birdstrikes_p2 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p2/"
        self.user_list_birdstrikes_p3 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p3/*"))
        self.user_location_birdstrikes_p3 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p3/"
        self.user_list_birdstrikes_p4 = np.sort(glob.glob(
            "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p4/*"))
        self.user_location_birdstrikes_p4 = "/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/birdstrikes_processed_interactions_p4/"

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
class OnlineSVM:
    def __init__(self, max_iter=1000):
        """
        Initializes the Online SVM model using SGDClassifier.
        """
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-3)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        for i in range(len(X_train)):
            self.model.partial_fit([X_train[i]], [y_train[i]], classes=['modify-1', 'modify-2' ,'modify-3', 'same'])


    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        # do online prediction predict , partial_fit, predict, partial_fit
        all_accuracies = []
        predictions=[]
        ground_truth=[]
        insight = defaultdict(list)
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            predictions.append(y_pred[0])
            ground_truth.append(y_test[i])
            #update the model telling the correct label
            if y_test[i] != y_pred[0]:
                self.model.partial_fit([X_test[i]], [y_test[i]])
            #all_accuracies.append(accuracy)
            if y_test[i] == y_pred[0]:
                insight[y_test[i]].append(1)
                all_accuracies.append(1)
            else:
                insight[y_test[i]].append(0)
                all_accuracies.append(0)


        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(all_accuracies), granular_prediction, predictions, ground_truth


    def evaluateOffline(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        predictions=[]
        ground_truth=[]
        #one shot prediction
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

def get_test_train_split(filename, thres):
        """
        Processes the data from the given file and stores it in the object memory.

        Inputs:
        - filename (str): path of the data file.
        - thres (float): threshold value to determine the end of the episode.

        Returns: None.
        """

        df = pd.read_csv(filename)
        threshold = int(len(df) * thres)
        mem_states = []
        mem_reward = []
        mem_action = []
        for index, row in df.iterrows():
            cur_state = row['State']
            action = row['Action']
            reward = row['Reward']
            mem_states.append(cur_state)
            mem_reward.append(reward)
            mem_action.append(action)

        # return train and data split by threshold states actions and rewards
        return mem_states[:threshold], mem_action[:threshold], mem_reward[:threshold], mem_states[threshold:], mem_action[threshold:], mem_reward[threshold:]
def run_experiment(user_list, dataset, task):
    """
    Run the experiment using Leave-One-Out Cross-Validation (LOOCV).
    """
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy', 'Algorithm'])

    user_list = shuffle(user_list, random_state=42)
    user_list = list(user_list)  # Convert numpy array to Python list

    y_true_all = []
    y_pred_all = []

    # Leave-One-Out Cross-Validation
    for i, test_user_log in enumerate(user_list):
        train_users = user_list[:i] + user_list[i+1:]  # All users except the ith one

        # Aggregate training data
        X_train = []
        y_train = []

        for user_log in train_users:
            env = environment_vizrec()
            env.process_data(user_log, 0)
            # Convert string representations of lists to actual lists
            states = [ast.literal_eval(state) for state in env.mem_states]
            X_train.extend(states)
            y_train.extend(env.mem_action)


        #get test users test and train data
        train_states, train_actions, _ ,test_states, test_actions,_= get_test_train_split(test_user_log,0.8)
        train_states = [ast.literal_eval(state) for state in train_states]
        X_train.extend(train_states)
        y_train.extend(train_actions)

        X_train = np.array(X_train)
        y_train = np.array(y_train)


       # Initialize and train the OnlineSVM model
        model = OnlineSVM()
        model.train(X_train, y_train)

        # Test on the left-out part of the test user
        user_name = os.path.basename(test_user_log).replace('_log.csv', '')


        # Convert string representations of lists to actual lists for test data
        X_test = np.array([ast.literal_eval(state) for state in test_states])
        y_test = np.array(test_actions)

        # Evaluate the model on the test data for this user
        accuracy, granularPredictions, pred, ground_truth = model.evaluate(X_test, y_test)

        # Store results
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
            'User': [user_name],
            'Accuracy': [accuracy],
            'Algorithm': ['OfflineOnlineSVM'],
            'GranularPredictions': [str(granularPredictions)],
            'Predictions': [str(pred)],
            'GroundTruth': [str(ground_truth)]

        })], ignore_index=True)



        #print(f"User {user_name} - Accuracy: {accuracy:.2f}")

    # Save results to
    # Define the output directory and file name
    directory = f"/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/Experiments_Folder/VizRec/{dataset}/{task}/PopulationThreshold"
    #os.makedirs(directory, exist_ok=True)
    output_file = f"{directory}/Online-SVM-Thresholded-Single-Model.csv"
    result_dataframe.to_csv(output_file, index=False)

    print(f"Dataset: {dataset} Task: {task} Algorithm: OnlineSVM, Average Accuracy: {result_dataframe['Accuracy'].mean()}")



def get_average_accuracy():
    """
    Get the average accuracy of the Online SVM model.
    """
    #for each dataset and task, read the results and calculate the average accuracy
    datasets = ['movies', 'birdstrikes']
    tasks = [ 'p1', 'p2','p3', 'p4']
    results = []
    for dataset in datasets:
        for task in tasks:
            directory = f"/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/Experiments_Folder/VizRec/{dataset}/{task}/PopulationThreshold"

            output_file = f"{directory}/Online-SVM-Thresholded-Single-Model.csv"
            result_df = pd.read_csv(output_file)
            print(f"Dataset: {dataset}, Task: {task}, Average Accuracy: {result_df['Accuracy'].mean()}")
            results.append(result_df['Accuracy'].mean())
    print(f"Overall Average Accuracy: {np.mean(results)}")


def delete_files(filename):
    """
    Delete the result files generated by the any Experiment.
    """
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2','p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            try:
                os.remove(f"Experiments_Folder/VizRec/{dataset}/{task}/{filename}")
            except FileNotFoundError:
              print(f"File not found: {filename}")

if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec()
            user_list = env.get_user_list(dataset, task)
            run_experiment(user_list, dataset, task)
            print(f"Done with {dataset} {task}")
    get_average_accuracy()
