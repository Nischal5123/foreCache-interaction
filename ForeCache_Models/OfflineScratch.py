import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDClassifier
from collections import defaultdict
import glob
import ast


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

from sklearn.svm import SVC  # Import the standard SVM

class OfflineSVM:
    """
    Offline SVM class for training using the full dataset at each threshold.
    """

    def __init__(self, actions):
        """
        Initializes the OfflineSVM object.
        """
        self.model = SVC(kernel='linear', C=1.0)  # Linear SVM
        self.actions = actions
        self.bestaction = defaultdict(lambda: random.choice(self.actions))
        self.reward = defaultdict(lambda: defaultdict(float))
        self.states = []

    def fit(self, X, y):
        """
        Trains the model using the entire training set at once.
        Args:
        - X (array-like): The feature matrix.
        - y (array-like): The target vector.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts the actions using the trained model.
        Args:
        - X (array-like): The feature matrix for prediction.
        Returns:
        - predictions (array-like): The predicted actions.
        """
        return self.model.predict(X)

    def update_best_action(self, state, action):
        """
        Updates the best action for a given state.
        Args:
        - state (str): The state.
        - action (str): The action taken.
        """
        self.bestaction[state] = action

    def offline_svm_driver(self, user, env, thres):
        """
        Trains the SVM using the offline (batch) method and tests it.
        Args:
        - user (str): The user's data file path.
        - env (environment_vizrec): The environment object.
        - thres (float): The threshold percentage to split training and testing data.
        Returns:
        - accuracy (float): The accuracy of predictions on the test set.
        """
        length = len(env.mem_action)
        threshold = int(length * thres)
        X_train, y_train = [], []

        # Prepare training data (from start to threshold)
        for i in range(threshold):
            state = env.mem_states[i]
            action = env.mem_action[i]
            self.update_best_action(state, action)
            # Convert state to features
            state_features = self.state_to_features(state)
            X_train.append(state_features)
            y_train.append(self.actions.index(action))

        #Ensure we have more than one class to train
        if len(set(y_train)) < 1:
            # Add a default example with an additional action to avoid issues
            X_train.append([0, 0, 0])  # Example dummy feature vector
            y_train.append(0)  # Different action to avoid single-class

        if len(set(y_train)) < 2:
            # Add a default example with an additional action to avoid issues
            X_train.append([0, 0, 0])  # Example dummy feature vector
            y_train.append((y_train[0] + 1) % len(self.actions))  # Different action to avoid single-class

        # Train the SVM on the full training set
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.fit(X_train, y_train)

        # Testing phase (from threshold to end of the data)
        accuracy = 0
        denom = 0
        for i in range(threshold, length):
            denom += 1
            state = env.mem_states[i]
            state_features = self.state_to_features(state)
            predicted_action = self.predict([state_features])[0]  # Predict the action
            actual_action = self.actions.index(env.mem_action[i])
            if predicted_action == actual_action:
                accuracy += 1

        # Return accuracy over the test set
        accuracy /= denom
        return accuracy

    def state_to_features(self, state):
        """
        Converts state to feature vector. Modify this function as needed.
        Args:
        - state (str): The state to convert.
        Returns:
        - features (np.array): The feature vector.
        """
        return np.array(ast.literal_eval(state))



def format_split_accuracy(accuracy_dict):
    main_states = ['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state = []
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.nanmean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None)  # No data for that state
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

def plot_predictions(y_true_all, y_pred_all, task, dataset, algorithm='OnlineSVM'):
    colors = {'same': 'blue', 'modify-1': 'green', 'modify-2': 'red', 'modify-3': 'purple'}
    matches = defaultdict(lambda: defaultdict(int))
    users = list(set(yt[2] for yt in y_true_all))

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

def save_data_to_csv(y_true_all, y_pred_all, task, dataset, algorithm='OnlineSVM'):
    data = []
    for yt, yp in zip(y_true_all, y_pred_all):
        data.append([yt[0], yt[1], yt[2], yp[0], yp[1]])

    df = pd.DataFrame(data, columns=['y_true', 'interaction_point', 'user', 'y_pred', 'pred_interaction_point'])
    directory = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
    os.makedirs(directory, exist_ok=True)
    df.to_csv(f"{directory}/{algorithm}_predictions_data.csv", index=False)


def get_overall_accuracy():
    #read the results from the csv files and get the overall accuracy
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    dataset_accuracy = []
    for dataset in datasets:
        task_accuracy = []
        for task in tasks:
            data = pd.read_csv("Experiments_Folder/Old-VizRec/{}/{}/OldOfflineSVM.csv".format(dataset, task))
            task_accuracy.append(data['Accuracy'].mean())
            print("Dataset: {}, Task: {}, OfflineSVM, {}".format(dataset, task, data['Accuracy'].mean()))
        dataset_accuracy.append(task_accuracy)
        print("Dataset: {}, OfflineSVM, {}".format(dataset, np.mean(task_accuracy)))
    print("Overall Accuracy: ", np.mean(dataset_accuracy))






if __name__ == "__main__":

    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    dataset_accuracy = []
    for dataset in datasets:
        task_accuracy = []
        for task in tasks:
            env = environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            result_dataframe = pd.DataFrame(
                columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])

            dataframe_users = []
            dataframe_threshold = []
            dataframe_learningrate = []
            dataframe_accuracy = []
            dataframe_discount = []
            dataframe_accuracy_per_state = []
            dataframe_algorithm = []

            user_list_2D = user_list_name
            total = 0
            y_accu_all = []
            for u in user_list_2D:
                y_accu = []
                threshold = [ 0.10, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                for thres in threshold:
                    env = environment_vizrec()
                    env.process_data(u, 0)
                    obj = OfflineSVM(actions=['same', 'modify-1', 'modify-2', 'modify-3'])
                    accu = obj.offline_svm_driver(u, env, thres)
                    accuracy_per_state = [0, 0, 0]  # Placeholder for state-based accuracy
                    total += accu
                    y_accu.append(accu)
                    del obj
                    dataframe_users.append(get_user_name(u))
                    dataframe_threshold.append(thres)
                    dataframe_learningrate.append(0)
                    dataframe_accuracy.append(accu)
                    dataframe_discount.append(0)
                    dataframe_accuracy_per_state.append(accuracy_per_state)
                    dataframe_algorithm.append("OfflineSVM")
                    env.reset(True, False)
                print(
                    "User ",
                    get_user_name(u),
                    " across all thresholds ",
                    "Global Accuracy: ",
                    np.nanmean(y_accu),
                )

                #plt.plot(threshold, y_accu, label=get_user_name(u), marker="*")
                y_accu_all.append(y_accu)
            # plt.yticks(np.arange(0.0, 1.0, 0.1))
            #
            # plt.xlabel("Threshold")
            # plt.ylabel("Accuracy")
            title = "OldOfflineSVM"
            # mean_y_accu = np.nanmean([element for sublist in y_accu_all for element in sublist])
            # plt.axhline(
            #     mean_y_accu,
            #     color="red",
            #     linestyle="--",
            #     label="Average: " + "{:.2%}".format(mean_y_accu),
            # )
            # plt.legend(loc="center left", bbox_to_anchor=(1, 0))
            # plt.title(title)
            # location = "TestFigures/" + title
            # plt.savefig(location, bbox_inches="tight")
            # plt.close()

            result_dataframe['User'] = dataframe_users
            result_dataframe['Threshold'] = dataframe_threshold
            result_dataframe['LearningRate'] = dataframe_learningrate
            result_dataframe['Discount'] = dataframe_discount
            result_dataframe['Accuracy'] = dataframe_accuracy
            result_dataframe['Algorithm'] = dataframe_algorithm
            result_dataframe['StateAccuracy'] = dataframe_accuracy_per_state
            result_dataframe.to_csv("Experiments_Folder/Old-VizRec/{}/{}/{}.csv".format(dataset, task, title), index=False)
    get_overall_accuracy()