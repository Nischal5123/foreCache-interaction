import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import misc
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


class WSLS:
    """
    WSLS (Win-Stay Lose-Switch) class implements the Win-Stay Lose-Switch algorithm for user modeling.
    """

    def __init__(self):
        """
        Initializes the WSLS object.
        """
        self.bestaction = defaultdict(
            lambda: defaultdict(str)
        )  # Initializes a dictionary with default values
        self.reward = defaultdict(
            lambda: defaultdict(float)
        )  # Initializes a dictionary with default values
        self.states = []  # Defines the possible states of the environment
        self.actions =['same', 'modify-1', 'modify-2', 'modify-3'] # Defines the possible actions of the agent
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
        action_space=['same', 'modify-1', 'modify-2', 'modify-3']
        action_space = [f for f in action_space if f != action]
        next_action = random.choice(action_space)
        return next_action

    def wslsDriver(self, user, env, thres):
        """
        Implements the Win-Stay Lose-Switch algorithm for a given user and environment.

        Args:
        - user (list): a list containing the data of a given user.
        - env (environment2): an environment object.
        - thres (float): the threshold value.

        Returns:
        - accuracy (float): the accuracy of the algorithm.
        """
        length = len(env.mem_action)
        threshold = int(length * thres)

        accuracy = 0
        denom = 0

        result = []
        accuracy=[]
        split_accuracy = defaultdict(list)
        for i in range(threshold + 1, length - 1):
            cur_action = self.bestaction[env.mem_states[i]]
            result.append(env.mem_states[i])
            result.append(cur_action)
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
            if self.bestaction[env.mem_states[i]] == env.mem_action[i]:
                accuracy.append(1)
                split_accuracy[env.mem_states[i]].append(1)
            else:
                split_accuracy[env.mem_states[i]].append(0)
                accuracy.append(0)
            denom += 1


        obj = misc.misc([])
        print("{}, {:.2f}, {}".format(obj.get_user_name(user), np.mean(accuracy), result))
        self.bestaction.clear()
        self.reward.clear()
        return np.mean(accuracy),split_accuracy

def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state

def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname

if __name__ == "__main__":

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
    user_list_2D = env.user_list_2D
    total = 0
    threshold = [0.1]
    obj2 = misc.misc([])
    y_accu_all = []
    for u in user_list_2D:
        y_accu = []
        threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for thres in threshold:
            env.process_data(u, 0)
            obj = WSLS()
            accu,state_accuracy = obj.wslsDriver(u, env, thres)
            accuracy_per_state = format_split_accuracy(state_accuracy)
            total += accu
            y_accu.append(accu)
            dataframe_users.append(get_user_name(u))
            dataframe_threshold.append(thres)
            dataframe_learningrate.append(0)
            dataframe_accuracy.append(accu)
            dataframe_discount.append(0)
            dataframe_accuracy_per_state.append(accuracy_per_state)
            dataframe_algorithm.append("WSLS")
            env.reset(True, False)
        print(
            "User ",
            get_user_name(u),
            " across all thresholds ",
            "Global Accuracy: ",
            np.mean(y_accu),
        )

        plt.plot(threshold, y_accu, label=get_user_name(u), marker="*")
        y_accu_all.append(y_accu)
    plt.yticks(np.arange(0.0, 1.0, 0.1))

    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    title = "wsls_all_3_actions"
    mean_y_accu = np.mean([element for sublist in y_accu_all for element in sublist])
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
    result_dataframe.to_csv("Experiments_Folder/VizRec/{}.csv".format(title), index=False)
