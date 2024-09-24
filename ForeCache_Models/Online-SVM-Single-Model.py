import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import warnings
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
        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            predictions.append(y_pred)
            ground_truth.append(y_test[i])
            accuracy = accuracy_score([y_test[i]], y_pred)
            self.model.partial_fit([X_test[i]], [y_test[i]])
            all_accuracies.append(accuracy)

        return np.mean(all_accuracies), predictions, ground_truth

    def evaluate2(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        #one shot prediction
        y_pred = self.model.predict(X_test)
        all_accuracies = accuracy_score(y_test, y_pred)

        return all_accuracies, y_pred

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

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Initialize and train the OnlineSVM model
        model = OnlineSVM()
        model.train(X_train, y_train)

        # Test on the left-out user
        user_name = os.path.basename(test_user_log).replace('_log.csv', '')
        env = environment_vizrec()
        env.process_data(test_user_log, 0)

        # Convert string representations of lists to actual lists for test data
        X_test = np.array([ast.literal_eval(state) for state in env.mem_states])
        y_test = np.array(env.mem_action)

        # Evaluate the model on the test data for this user
        accuracy, pred, ground_truth = model.evaluate(X_test, y_test)

        # Store results
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
            'User': [user_name],
            'Accuracy': [accuracy],
            'Algorithm': ['OnlineSVM'],
            'Granular_Predictions': [None],
            'Predictions': str(pred),
            'Ground_Truth': str(ground_truth)

        })], ignore_index=True)



        #print(f"User {user_name} - Accuracy: {accuracy:.2f}")

    # Save results to CSV
    result_dataframe.to_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/Online-SVM-Single-Model.csv", index=False)
    plot_predictions(y_true_all, y_pred_all, task, dataset)
    print(f"Dataset: {dataset} Task: {task} Algorithm: OnlineSVM, Average Accuracy: {result_dataframe['Accuracy'].mean()}")

def plot_predictions(y_true_all, y_pred_all, task, dataset):
    """
    Plot the predictions for all users.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true_all, label='True', alpha=0.6)
    plt.plot(y_pred_all, label='Predicted', alpha=0.6)
    plt.legend()
    plt.title(f"True vs Predicted Actions for Task {task}")
    plt.savefig(f"Experiments_Folder/VizRec/{dataset}/{task}/predictions_plot.png")
    plt.close()

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
            result_df = pd.read_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/Online-SVM-Single-Model.csv")
            print(f"Dataset: {dataset}, Task: {task}, Average Accuracy: {result_df['Accuracy'].mean()}")
            results.append(result_df['Accuracy'].mean())
    print(f"Overall Average Accuracy: {np.mean(results)}")


if __name__ == "__main__":
    # datasets = ['movies', 'birdstrikes']
    # tasks = ['p1', 'p2','p3', 'p4']
    # for dataset in datasets:
    #     for task in tasks:
    #         env = environment_vizrec()
    #         user_list = env.get_user_list(dataset, task)
    #         run_experiment(user_list, dataset, task)
    #         print(f"Done with {dataset} {task}")
    get_average_accuracy()
