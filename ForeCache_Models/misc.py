#contains all the miscellaneous functions for running 
import pdb
import random
import SARSA
import numpy as np
import matplotlib.pyplot as plt 
import sys
import plotting
import environment2
from tqdm import tqdm
from random import randint
import TDLearning
from collections import Counter


class misc:
    def __init__(self, users):
        self.discount_h = [0.05, 0.1, 0.2,0.5, 0.8, 0.9]
        self.alpha_h = [0.0001,0.001,0.01, 0.1, 0.5]
        self.epsilon_h = [0.95, 0.5, 0.8,0.3,0.05]
        self.threshold_h = []
        self.prog = users * len(self.epsilon_h) * len(self.alpha_h) * len(self.discount_h) * len(self.threshold_h)

    def get_user_name(self, url):
        string = url.split('\\')
        fname = string[len(string) - 1]
        uname = ((fname.split('userid_')[1]).split('.'))[0]
        return uname

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
        best_discount = best_alpha = best_eps = -1
        e = a = d = 0
        pp = 10
        # with tqdm(total = self.prog) as pbar:
        y_accu_all=[]
        for user in users_hyper:
            # print(user)
            max_accu = -1
            self.threshold_h = self.get_threshold(env, user)
            y_accu = []
            #   self.threshold_h =[env.get_threshold(user)]
            for thres in self.threshold_h:
                max_accu_thres = -1
                env.process_data(user, thres)
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            accu = 0
                            for epiepi in range(pp):
                                if algorithm == 'qlearning':
                                    obj = TDLearning.TDLearning()
                                    # pdb.set_trace()
                                    Q, stats = obj.q_learning(user, env, epoch, dis, alp, eps)
                                else:
                                    obj = SARSA.TD_SARSA()
                                    Q, stats = obj.sarsa(user, env, epoch, dis, alp, eps)

                                test_accuracy, stats = obj.test(env, Q, dis, alp, eps)
                                accu += test_accuracy
                            # print(accu/20)
                            if max_accu_thres < accu:
                                max_accu = accu
                                best_eps = eps
                                best_alpha = alp
                                best_discount = dis
                            max_accu_thres = max(max_accu_thres, accu)
                            # pbar.update(1)
                env.reset(True, False)
                y_accu.append(round(max_accu_thres / pp, 2))
                max_accu = max(max_accu_thres, max_accu)
                y_accu_all.append(y_accu)
            plt.plot(self.threshold_h, y_accu, label=self.get_user_name(user), marker='*')
            mean_y_accu = np.mean([element for sublist in y_accu_all for element in sublist])
            plt.axhline(mean_y_accu, color='red', linestyle='--', )
            print("{}, {:.2f}, {}, {}, {}".format(self.get_user_name(user), max_accu / pp, best_eps, best_discount,
                                                  best_alpha))
            e += best_eps
            d += best_discount
            a += best_alpha
        plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        plt.yticks(np.arange(0.0, 1.0, 0.1))
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        title = algorithm  + str(randint(100, 999))
        # pdb.set_trace()
        plt.title(title)
        location = 'figures/' + title
        plt.savefig(location, bbox_inches='tight')
        plt.close()

        # return best_eps, best_discount, best_alpha
        print("best epsilon ", e, ",best_discount ", d, ",best_alpha ",
              a)
        # print("best epsilon ", e / len(users_hyper), ",best_discount ",d / len(users_hyper),",best_alpha ",a / len(users_hyper))
        # return e / len(users_hyper), d / len(users_hyper), a / len(users_hyper)

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