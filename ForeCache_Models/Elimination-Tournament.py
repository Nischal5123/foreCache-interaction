import environment_vizrec
import numpy as np
from collections import defaultdict
import random
import misc
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random

import random

class ComparisonRunner:
    def __init__(self):
        pass

    def get_action_success(self, action, step):
        return 1 if self.env_mem_action[step] == action else -1

    def run_comparison(self, N, M, current_action1, current_action2, start):
        counters = {current_action1: 0, current_action2: 0}
        current_action = current_action1
        accuracy = []

        for step in range(start, N):
            # Stop the comparison with probability 1/N or if counter reaches M or -M
            if random.random() < 1 / N or any(abs(counter) >= M for counter in counters.values()):
                winner = max(counters, key=counters.get)
                return winner, step - 1, accuracy

            success1 = self.get_action_success(current_action, step)
            # Append 1 for success and 0 for failure to accuracy list
            accuracy.append(1 if success1 == 1 else 0)

            success2 = -1 * success1
            counters[current_action] += success1
            other_action = current_action2 if current_action == current_action1 else current_action1
            counters[other_action] += success2

            # Alternate to the other action without affecting counters
            current_action = other_action

        # Declare the winner based on maximum counter value
        winner = max(counters, key=counters.get)

        return winner, step, accuracy

    def run_tournament(self, u, env, thres, actions, M ,N=None):
        self.env_mem_action = env.mem_action
        length = len(env.mem_action)
        N=length-1
        threshold = int(length * thres)
        total_accuracy = []
        start = threshold + 1  # Use the last value of start from the previous iteration

        while len(actions) > 1 and start < length - 1:
            # Alternate between actions
            current_action1 = random.choice(actions)
            current_action2 = random.choice([action for action in actions if action != current_action1])
            result, start, accuracy = self.run_comparison(N, M, current_action1, current_action2, start)
            total_accuracy += accuracy

            if result:
                # Run another comparison excluding the loser
                other_actions = [action for action in actions if action != result]
                actions = other_actions
            else:
                print('No winner declared.')

        # Check if the tournament reached the end
        if start != length - 1:
            for k in range(start + 1, length - 1):
                success = self.get_action_success(result, k)
                # Append 1 for success and 0 for failure to total_accuracy list
                total_accuracy.append(1 if success == 1 else 0)
        print(len(total_accuracy) , length - threshold)
        #assert len(total_accuracy) == length - threshold

        return np.mean(total_accuracy)


def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

if __name__ == '__main__':
    M = 50
    N=1000
    #actions = ['same', 'modify']
    actions=['same', 'modify-1', 'modify-2', 'modify-3']
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
    y_accu_all = []
    for u in user_list_2D:
        y_accu = []
        threshold = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for thres in threshold:
            env.process_data(u, 0)
            obj = ComparisonRunner()
            accu = obj.run_tournament(u, env, thres, actions, M, N)
            y_accu.append(accu)
            dataframe_users.append(get_user_name(u))
            dataframe_threshold.append(thres)
            dataframe_learningrate.append(0)
            dataframe_accuracy.append(accu)
            dataframe_discount.append(0)
            dataframe_accuracy_per_state.append(0)
            dataframe_algorithm.append("Elimination")
            env.reset(True, False)
        print(
            "User ",
            get_user_name(u),
            " across all thresholds ",
            "Global Accuracy: ",
            np.mean(y_accu),
        )
    title='Eliminiation-Tournament'
    result_dataframe['User'] = dataframe_users
    result_dataframe['Threshold'] = dataframe_threshold
    result_dataframe['LearningRate'] = dataframe_learningrate
    result_dataframe['Discount'] = dataframe_discount
    result_dataframe['Accuracy'] = dataframe_accuracy
    result_dataframe['Algorithm'] = dataframe_algorithm
    result_dataframe['StateAccuracy'] = dataframe_accuracy_per_state
    result_dataframe.to_csv("Experiments_Folder/VizRec/{}.csv".format(title), index=False)




