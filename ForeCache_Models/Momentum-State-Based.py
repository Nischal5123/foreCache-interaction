import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import misc
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


class Momentum:
    """
    Momentum
    """

    def __init__(self):
        """
        Initializes the Momentum object.
        """
        self.bestaction = defaultdict(
            lambda: defaultdict(str)
        )  # Initializes a dictionary with default values
        self.reward = defaultdict(
            lambda: defaultdict(float)
        )  # Initializes a dictionary with default values
        self.states = []  # Defines the possible states of the environment
        self.actions =['same', 'modify-1', 'modify-2','modify-3'] # Defines the possible actions of the agent
        #self.actions = ['same', 'modify']  # Defines the possible actions of the agent
        self.bestaction = defaultdict(lambda: self.take_random_action('',''))
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
        #action_space = ['same', 'modify-x', 'modify-y', 'modify-z', 'modify-x-y', 'modify-y-z', 'modify-x-z','modify-x-y-z']
        #action_space=['same', 'modify']
        action_space = [f for f in self.actions if f != action]
        next_action = random.choice(action_space)
        return next_action

    def MomentumDriver(self, user, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(threshold):
            self.bestaction[env.mem_states[i]] = env.mem_action[i]

        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        for i in range(threshold, length):
            denom += 1
            try:  # Finding the last action in the current state
                candidate = self.bestaction[env.mem_states[i]]
            except KeyError:  # Randomly picking an action if the current state is new
                candidate = random.choice(self.actions)

            if candidate == env.mem_action[i]:
                accuracy += 1

        accuracy /= denom
        return accuracy

def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.nanmean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

if __name__ == "__main__":

    datasets = ['movies','birdstrikes']
    tasks = ['p1','p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
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

            env = environment_vizrec.environment_vizrec()
            user_list_2D = user_list_name
            total = 0
            obj2 = misc.misc([])
            y_accu_all = []
            for u in user_list_2D:
                y_accu = []
                threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                for thres in threshold:
                    env.process_data(u, 0)
                    obj = Momentum()
                    accu= obj.MomentumDriver(u, env, thres)
                    accuracy_per_state = [0,0,0]
                    total += accu
                    y_accu.append(accu)
                    dataframe_users.append(get_user_name(u))
                    dataframe_threshold.append(thres)
                    dataframe_learningrate.append(0)
                    dataframe_accuracy.append(accu)
                    dataframe_discount.append(0)
                    dataframe_accuracy_per_state.append(accuracy_per_state)
                    dataframe_algorithm.append("Momentum")
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
            title = "Momentum"
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
            result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset,task,title), index=False)
