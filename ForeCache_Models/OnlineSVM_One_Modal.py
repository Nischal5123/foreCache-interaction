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

class OnlineSVM:
    """
    Online SVM with Partial Fit
    """

    def __init__(self, actions):
        """
        Initializes the OnlineSVM object.
        """
        self.model = SGDClassifier(loss='hinge', alpha=1e-3, max_iter=1000)  # SVM with hinge loss
        self.actions = actions
        self.bestaction = defaultdict(lambda: random.choice(self.actions))
        self.reward = defaultdict(lambda: defaultdict(float))
        self.states = []

    def partial_fit(self, X, y, training=False):
        """
        Performs a partial fit on the given data.

        Args:
        - X (array-like): The feature matrix.
        - y (array-like): The target vector.
        """
        if training:
            self.model.partial_fit(X, y, classes=np.arange(len(self.actions)))
        else:
            self.model.partial_fit(X, y)

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

    def online_svm_driver(self, user, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        X_train, y_train = [], []

        for i in range(threshold+1):
            state = env.mem_states[i]
            action = env.mem_action[i]
            self.update_best_action(state, action)
            # Prepare data for partial fit
            state_features = self.state_to_features(state)
            #add some placeholder data state [0,0,0] + all actions

            X_train.append(state_features)
            y_train.append(self.actions.index(action))
        # #handle less than 2 classes
        # if len(set(y_train)) < 2:
        #     #add one action to the training set other than the current action
        #     X_train.append([0, 0, 0])
        #     y_train.append(y_train[0] + 1)

        # Train the model with partial fit
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # if len(X_train) <1 :  # If no data then we add atleast one example with the most common action : only happens for 5 users when threshold is 0.1
        #     print("########################### No data for user: ", user)
        #     X_train = np.array([[0,0,0]])
        #     y_train = np.array([0])



        self.partial_fit(X_train, y_train, training=True)

        # Checking accuracy on the remaining data
        accuracy = 0
        denom = 0
        insight = defaultdict(list)

        for i in range(threshold, length):
            denom += 1
            state = env.mem_states[i]
            state_features = self.state_to_features(state)
            candidate = self.predict([state_features])[0]
            #fit
            self.partial_fit([state_features], [self.actions.index(env.mem_action[i])] )
            candidate_action = self.actions[candidate]
            if candidate_action == env.mem_action[i]:
                accuracy += 1
            insight[env.mem_action[i]].append(candidate_action == env.mem_action[i])

        accuracy /= denom
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        return accuracy, granular_prediction

    def state_to_features(self, state):
        """
        Converts state to feature vector. Implement as needed.
        """
        return np.array(ast.literal_eval(state))



def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname



def get_overall_accuracy():
    #read the results from the csv files and get the overall accuracy
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    dataset_accuracy = []
    for dataset in datasets:
        task_accuracy = []
        for task in tasks:
            data = pd.read_csv("Experiments_Folder/VizRec/{}/{}/OnlineSVM.csv".format(dataset, task))
            task_accuracy.append(data['Accuracy'].mean())
            print("Dataset: {}, Task: {}, OnlineSVM, {}".format(dataset, task, data['Accuracy'].mean()))
        dataset_accuracy.append(task_accuracy)
        print("Dataset: {}, OnlineSVM, {}".format(dataset, np.mean(task_accuracy)))
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
                threshold = [ 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                for thres in threshold:
                    env = environment_vizrec()
                    env.process_data(u, 0)
                    obj = OnlineSVM(actions=['same', 'modify-1', 'modify-2', 'modify-3'])
                    accu, granular_predictions = obj.online_svm_driver(u, env, thres)
                    total += accu
                    y_accu.append(accu)
                    del obj
                    dataframe_users.append(get_user_name(u))
                    dataframe_threshold.append(thres)
                    dataframe_learningrate.append(0)
                    dataframe_accuracy.append(accu)
                    dataframe_discount.append(0)
                    dataframe_accuracy_per_state.append(granular_predictions)
                    dataframe_algorithm.append("OnlineSVM")
                    env.reset(True, False)
                print(
                    "User ",
                    get_user_name(u),
                    " across all thresholds ",
                    "Global Accuracy: ",
                    np.nanmean(y_accu),
                )

                plt.plot(threshold, y_accu, label=get_user_name(u), marker="*")
                y_accu_all.append(y_accu)
            plt.yticks(np.arange(0.0, 1.0, 0.1))

            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            title = "OnlineSVM"
            mean_y_accu = np.nanmean([element for sublist in y_accu_all for element in sublist])
            plt.axhline(
                mean_y_accu,
                color="red",
                linestyle="--",
                label="Average: " + "{:.2%}".format(mean_y_accu),
            )
            plt.legend(loc="center left", bbox_to_anchor=(1, 0))
            plt.title(title)
            location = "TestFigures/" + title
            plt.savefig(location, bbox_inches="tight")
            plt.close()

            result_dataframe['User'] = dataframe_users
            result_dataframe['Threshold'] = dataframe_threshold
            result_dataframe['LearningRate'] = dataframe_learningrate
            result_dataframe['Discount'] = dataframe_discount
            result_dataframe['Accuracy'] = dataframe_accuracy
            result_dataframe['Algorithm'] = dataframe_algorithm
            result_dataframe['StateAccuracy'] = dataframe_accuracy_per_state
            result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset, task, title), index=False)
    get_overall_accuracy()