#contains all the miscellaneous functions for running 
import pandas as pd
import SARSA
import numpy as np
import matplotlib.pyplot as plt 
import json
import TDLearning
from collections import Counter


class misc:
    def __init__(self, users,hyperparam_file='sampled-hyperparameters-config.json'):
        """
        Initializes the misc class.
        Parameters:
    - users: List of users
    - hyperparam_file: File path to the hyperparameters JSON file
    """
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        # Extract hyperparameters from JSON file
        self.discount_h =hyperparams['gammas']
        self.alpha_h = hyperparams['learning_rates']
        self.epsilon_h = hyperparams['epsilon']
        self.threshold_h = hyperparams['threshold']
        self.main_states=['Foraging', 'Navigation', 'Sensemaking']
        self.prog = users * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h) * len(self.threshold_h)

    def get_user_name(self, url):
        """
            Extracts the username from the URL.

            Parameters:
            - url: URL string

            Returns:
            - uname: Username extracted from the URL
            """
        string = url.split('\\')
        fname = string[len(string) - 1]
        uname = fname.rstrip('.csv')
        return uname

    def format_split_accuracy(self, accuracy_dict):
        """
            Formats the accuracy per state.

            Parameters:
            - accuracy_dict: Dictionary containing accuracy values for each state

            Returns:
            - accuracy_per_state: List of accuracy values per state
            """
        accuracy_per_state = []
        for state in self.main_states:
            if accuracy_dict[state]:
                accuracy_per_state.append(np.mean(accuracy_dict[state]))
            else:
                accuracy_per_state.append(0)
        return accuracy_per_state

    def get_threshold(self, env, user):
        """
            Calculates the threshold.

            Parameters:
            - env: Environment object
            - user: User data

            Returns:
            - proportions: List of threshold proportions
            """
        env.process_data(user, 0)
        counts = Counter(env.mem_roi)
        proportions = []
        total_count = len(env.mem_roi)

        for i in range(1, max(counts.keys()) + 1):
            current_count = sum(counts[key] for key in range(1, i + 1))
            proportions.append(current_count / total_count)
        return proportions[:-1]

    def hyper_param(self, env, users_hyper, algorithm, epoch):
        """
            Performs hyperparameter optimization.

            Parameters:
            - env: Environment object
            - users_hyper: List of user data
            - algorithm: Algorithm name ('QLearn' or 'SARSA')
            - epoch: Number of epochs

            Returns:
            None
            """
        result_dataframe = pd.DataFrame(
            columns=['Algorithm','User','Epsilon', 'Threshold', 'LearningRate', 'Discount','Accuracy','StateAccuracy','Reward'])
        best_discount = best_alpha = best_eps = -1
        pp = 10
        y_accu_all=[]
        for user in users_hyper:

            y_accu = []

            for thres in self.threshold_h:
                max_accu_thres = -1
                env.process_data(user, thres)
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for epiepi in range(pp):
                                if algorithm == 'QLearn':
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
                                    best_q=Q
                                    best_obj=obj
                                max_accu_thres = max(max_accu_thres, train_accuracy)
                print("Top Training Accuracy: {}, Threshold: {}".format(max_accu_thres, thres))
                test_accuracy, stats, split_accuracy, reward = best_obj.test(env, best_q, best_discount, best_alpha, best_eps)
                accuracy_per_state=self.format_split_accuracy(split_accuracy)

                print(
                    "Algorithm:{} , User:{}, Threshold: {}, Test Accuracy:{},  Epsilon:{}, Alpha:{}, Discount:{}, Split_Accuracy:{}".format(
                        algorithm,
                        self.get_user_name(user), thres, test_accuracy, best_eps, best_alpha,
                        best_discount,accuracy_per_state))
                print("Action choice: {}".format(Counter(stats)))

                #book-keeping
                result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                    'User': [self.get_user_name(user)],
                    'Threshold': [thres],
                    'Epsilon': [best_eps],
                    'LearningRate': [best_alpha],
                    'Discount': [best_discount],
                    'Accuracy': [test_accuracy],
                    'StateAccuracy': [accuracy_per_state],
                    'Algorithm': [algorithm],
                    'Reward': [reward]
                })], ignore_index=True)


                #end book-keeping

                y_accu.append(test_accuracy)
                y_accu_all.append(y_accu)

                ###move to new threshold:
                env.reset(True, False)

            plt.plot(self.threshold_h, y_accu, label=self.get_user_name(user), marker='*')
        mean_y_accu = np.mean([element for sublist in y_accu_all for element in sublist])
        plt.axhline(mean_y_accu, color='red', linestyle='--',label="Average: "+ "{:.2%}".format(mean_y_accu) )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        title = algorithm  + "all_3_actions"
        # pdb.set_trace()
        plt.title(title)
        location = 'figures/' + title
        plt.savefig(location, bbox_inches='tight')
        result_dataframe.to_csv("Experiments_Folder\\" + title + ".csv", index=False)
        plt.close()




