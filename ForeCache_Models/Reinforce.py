import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment2
import plotting
from collections import Counter
import misc

import multiprocessing
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# # Hyperparameters
# learning_rate = 0.0002
# gamma = 0.99


class Policy(nn.Module):
    def __init__(self,learning_rate,gamma):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 2)
        self.gamma=gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class Reinforce():
    def __init__(self,env,learning_rate,gamma):
        self.env = env
        self.learning_rate, self.gamma = learning_rate, gamma
        self.pi = Policy(self.learning_rate, self.gamma)



    def main(self):
        print_interval =10
        score=0.0
        for n_epi in range(50):
            s = self.env.reset()
            if s == 'Sensemaking':
                s=[1,0,0]
            elif s== 'Foraging':
                s=[0,1,0]
            else:
                s=[0,0,1]
            s=np.array(s)
            done = False
            actions =[]
            while not done:  # CartPole-v1 forced to terminates at 500 step.
                prob = self.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, info = self.env.step(s,a.item(),False)
                self.pi.put_data((r, prob[a]))
                s = s_prime
                if s == 'Sensemaking':
                    s = [1, 0, 0]
                elif s == 'Foraging':
                    s = [0, 1, 0]
                else:
                    s = [0, 0, 1]
                s = np.array(s)
                score += r
                actions.append(a.item())

            self.pi.train_net()

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {}, actions :{}".format(n_epi, score / print_interval, Counter(actions)))
            score = 0.0

        test_print_interval=20
        test_accuracies=[]
        for n_epi in range(1):
            s = self.env.reset(all=False , test=True)
            if s == 'Sensemaking':
                s=[1,0,0]
            elif s== 'Foraging':
                s=[0,1,0]
            else:
                s=[0,0,1]
            s=np.array(s)
            done = False
            predictions =[]
            actions=[]
            while not done:  # CartPole-v1 forced to terminates at 500 step.
                prob = self.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                s_prime, r, done, info = self.env.step(s,a.item(),True)
                #pi.put_data((r, prob[a]))
                s = s_prime
                if s == 'Sensemaking':
                    s = [1, 0, 0]
                elif s == 'Foraging':
                    s = [0, 1, 0]
                else:
                    s = [0, 0, 1]
                s = np.array(s)
                score += r
                predictions.append(info)
                actions.append(a.item())


            print("TEST: # of episode :{}, accuracy : {}, actions: {}".format(n_epi, np.mean(predictions),Counter(actions)))
            score = 0.0
            test_accuracies.append(np.mean(predictions))
        return np.mean(test_accuracies)


def get_threshold(env, user):
    env.process_data(user, 0)
    counts = Counter(env.mem_roi)
    proportions = []
    total_count = len(env.mem_roi)

    for i in range(1, max(counts.keys()) + 1):
        current_count = sum(counts[key] for key in range(1, i + 1))
        proportions.append(current_count / total_count)
    return proportions[:-1]


def user_set(e,user_list,exp):
    result_dataframe = pd.DataFrame(
        columns=['User', 'Threshold', 'LearningRate', 'Discount'])

    dataframe_users = []
    dataframe_threshold = []
    dataframe_learningrate = []
    dataframe_discount = []
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
    gammas = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

    aggregate_plotter = plotting.plotter(None)
    y_accu_all = []
    miscelleneous = misc.misc([])


    for u in user_list:
        threshold_h = get_threshold(e, u)
        plotter = plotting.plotter(threshold_h)
        y_accu = []
        user_name = miscelleneous.get_user_name(u)
        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0

            for learning_rate in learning_rates:
                for gamma in gammas:
                    env = environment2.environment2()
                    env.process_data(u, thres)
                    agent = Reinforce(env,learning_rate,gamma)
                    accuracies = agent.main()

                    if accuracies > max_accu:
                        max_accu=accuracies
                        best_learning_rate=learning_rate
                        best_gamma=gamma


            y_accu.append(max_accu)
            dataframe_users.append(user_name)
            dataframe_threshold.append(thres)
            dataframe_learningrate.append(best_learning_rate)
            dataframe_discount.append(best_gamma)
            print(
                "# User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}".format(user_name, thres, max_accu,
                                                                                            best_learning_rate,
                                                                                            best_gamma))
        plotter.plot_main(y_accu, user_name)
        y_accu_all.append(y_accu)
    title = "reinforce" + exp
    aggregate_plotter.aggregate(y_accu_all, title)
    result_dataframe['User'] = dataframe_users
    result_dataframe['Threshold'] = dataframe_threshold
    result_dataframe['LearningRate'] = dataframe_learningrate
    result_dataframe['Discount'] = dataframe_discount
    result_dataframe.to_csv("data/NDSI-2D\\" + title + ".csv", index=False)

if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    user_list_experienced = np.array(
        ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
    user_list_first_time = np.setdiff1d(user_list_2D, user_list_experienced)
    p111 = multiprocessing.Process(target=user_set, args=(env,user_list_first_time[:6],'_first_time'))
    p112 = multiprocessing.Process(target=user_set, args=(env, user_list_experienced[:6],'_experienced'))
    p111.start()
    p112.start()

    p111.join()
    p112.join()



