#contains all the miscellaneous functions for running 
import SARSA_new as SARSA
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import sys
import plotting
import environment2
from tqdm import tqdm
from random import randint
import TDLearning_new as TDLearning
from collections import Counter
# import numba

class misc:
    def __init__(self, users):
        self.discount_h = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
        self.alpha_h = [0.0001, 0.001, 0.01, 0.1, 0.5]
        self.epsilon_h = [0.95, 0.5, 0.8, 0.3, 0.05]
        self.threshold_h = []
        self.main_states = ['Foraging', 'Navigation', 'Sensemaking']
        self.prog = users * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h) * len(self.threshold_h)

    def get_user_name(self, url):
        string = url.split('\\')
        fname = string[len(string) - 1]
        uname = fname.rstrip('.csv')
        return uname

    def format_split_accuracy(self,accuracy_dict):
        accuracy_per_state=[]
        for state in self.main_states:
            if accuracy_dict[state]:
                accuracy_per_state.append(np.mean(accuracy_dict[state]))
            else:
                accuracy_per_state.append(0)
        return accuracy_per_state

    def get_threshold(self, env, user):
        env.process_data(user, 0)
        counts = Counter(env.mem_roi)
        proportions = []
        total_count = len(env.mem_roi)

        for i in range(1, max(counts.keys()) + 1):
            current_count = sum(counts[key] for key in range(1, i + 1))
            proportions.append(current_count / total_count)
        return proportions[:-1]

    def hyper_param(self, env, users_hyper, algorithm, epoch):

        result_dataframe = pd.DataFrame(
            columns=['User', 'Epsilon', 'Threshold', 'LearningRate', 'Discount', 'Accuracy', 'StateAccuracy'])

        dataframe_users = []
        dataframe_epsilon = []
        dataframe_threshold = []
        dataframe_learningrate = []
        dataframe_discount = []
        dataframe_accuracy = []
        dataframe_accuracy_per_state = []

        best_discount = best_alpha = best_eps = -1
        pp = 1
        y_accu_all = []
        for user in users_hyper:
            # print(user)
            max_accu = -1
            self.threshold_h = self.get_threshold(env, user)
            self.threshold_h = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            y_accu = []
            #   self.threshold_h =[env.get_threshold(user)]
            for thres in self.threshold_h:
                max_accu_thres = -1
                env.process_data(user, thres)
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for epiepi in range(pp):
                                if algorithm == 'qlearning':
                                    obj = TDLearning.TDLearning()
                                    Q, train_accuracy = obj.q_learning(user, env, epoch, dis, alp, eps)
                                else:
                                    obj = SARSA.TD_SARSA()
                                    Q, train_accuracy = obj.sarsa(user, env, epoch, dis, alp, eps)

                                if max_accu_thres < train_accuracy:
                                    max_accu_thres = train_accuracy
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                    best_Q = Q
                                    best_obj = obj
                                max_accu_thres = max(max_accu_thres, train_accuracy)
                print("Top Training Accuracy: {}, Threshold: {}".format(max_accu_thres, thres))
                test_accuracy, stats, split_accuracy = best_obj.test(env, best_Q, best_discount, best_alpha, best_eps)
                accuracy_per_state = self.format_split_accuracy(split_accuracy)
                #assert(abs(np.mean(accuracy_per_state)-test_accuracy)<0.1)
                print(
                    "Algorithm:{} , User:{}, Threshold: {}, Test Accuracy:{},  Epsilon:{}, Alpha:{}, Discount:{}, Split_Accuracy:{}".format(
                        algorithm,
                        self.get_user_name(user), thres, test_accuracy, best_eps, best_alpha,
                        best_discount, accuracy_per_state))

                # book-keeping
                dataframe_users.append(self.get_user_name(user))
                dataframe_epsilon.append(best_eps)
                dataframe_threshold.append(thres)
                dataframe_learningrate.append(best_alpha)
                dataframe_discount.append(best_discount)
                dataframe_accuracy.append(test_accuracy)
                dataframe_accuracy_per_state.append(accuracy_per_state)
                # end book-keeping

                y_accu.append(test_accuracy)
                y_accu_all.append(y_accu)

                ###move to new threshold:
                env.reset(True, False)

            plt.plot(self.threshold_h, y_accu, label=self.get_user_name(user), marker='*')
        mean_y_accu = np.mean([element for sublist in y_accu_all for element in sublist])
        plt.axhline(mean_y_accu, color='red', linestyle='--', label="Average: " + "{:.2%}".format(mean_y_accu))

        result_dataframe['User'] = dataframe_users
        result_dataframe['Threshold'] = dataframe_threshold
        result_dataframe['LearningRate'] = dataframe_learningrate
        result_dataframe['Epsilon'] = dataframe_epsilon
        result_dataframe['Discount'] = dataframe_discount
        result_dataframe['Accuracy'] = dataframe_accuracy
        result_dataframe['StateAccuracy'] = dataframe_accuracy_per_state
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        title = algorithm + "_decaying_all"
        # pdb.set_trace()
        plt.title(title)
        location = 'figures/' + title
        plt.savefig(location, bbox_inches='tight')
        result_dataframe.to_csv("Experiments_Folder\\" + title + ".csv", index=False)
        plt.close()

    def plot(self, x_labels, y, title):
        x = []
        for i in range(0, len(x_labels)):
            x.append(i)
        plt.xticks(x, x_labels)
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.figure(figsize=(20, 10))
        plt.plot(x, y)
        plt.xlabel('Users')
        plt.ylabel('Accuracy')
        plt.title(title)
        location = 'figures/' + title
        plt.savefig(location)
        plt.close()
        # plt.show()

    def run_stuff(self, env, users, epoch, title, best_eps, best_discount, best_alpha, algo):
        x = []
        y = []
        for u in users:
            # print(u)
            sum = 0
            x.append(self.get_user_name(u))
            for episodes in tqdm(range(epoch)):
                env.process_data(u, self.threshold_h[0])
                if algo == 'sarsa':
                    # print("S")
                    obj = TDLearning.TDLearning()
                    Q, stats = obj.q_learning(u, env, 50, best_discount, best_alpha, best_eps)
                else:
                    # print("Q")
                    obj = SARSA.TD_SARSA()
                    Q, stats = obj.sarsa(u, env, 50, best_discount, best_alpha, best_eps)

                accu = obj.test(env, Q, best_discount, best_alpha, best_eps)
                sum += accu
                # print("{} {}".format(u, accu))
                env.reset(True, False)
                # pdb.set_trace()
            print("{} {} {}".format(algo, u, round(sum / epoch, 2)))
            y.append(round(sum / epoch, 2))
        self.plot(x, y, title)