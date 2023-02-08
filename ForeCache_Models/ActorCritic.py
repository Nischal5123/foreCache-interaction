import environment2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import plotting
from collections import Counter
import misc
import pandas as pd
import multiprocessing
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Hyperparameters
# learning_rate = 0.0002
# gamma = 0.99
# n_rollout = 30


class ActorCritic(nn.Module):
    def __init__(self,learning_rate,gamma):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(
            a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(
            s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

class Agent():
    def __init__(self, env,learning_rate,gamma,num_rollouts=10):
        self.env = env
        self.learning_rate, self.gamma, self.n_rollout=learning_rate,gamma,num_rollouts

    def train(self):
        model = ActorCritic(self.learning_rate, self.gamma)
        print_interval = 10
        score = 0.0
        all_predictions = []
        for n_epi in range(50):
            done = False
            s = self.env.reset()
            if s == 'Sensemaking':
                s = [1, 0, 0]
            elif s == 'Foraging':
                s = [0, 1, 0]
            else:
                s = [0, 0, 1]
            s = np.array(s)
            predictions = []
            actions = []
            while not done:
                for t in range(self.n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    actions.append(a)
                    s_prime, r, done, info = self.env.step(s, a, False)
                    predictions.append(info)
                    ######################

                    if s_prime == 'Sensemaking':
                        s_prime = [1, 0, 0]
                    elif s_prime == 'Foraging':
                        s_prime = [0, 1, 0]
                    else:
                        s_prime = [0, 0, 1]
                    s_prime = np.array(s_prime)

                    ################
                    model.put_data((s, a, r, s_prime, done))

                    s = s_prime

                    score += r

                    if done:
                        break

                model.train_net()

            # if n_epi % print_interval == 0 and n_epi != 0:
            #   print("# of episode :{}, avg score : {:.1f}, accuracy: {:.1f} , actions{}".format(n_epi, score / print_interval,np.mean(predictions),Counter(actions)))
            score = 0.0
            all_predictions.append(np.mean(predictions))
        print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
        return model, np.mean(predictions)  # return last episodes accuracyas training accuracy

    def test(self,model):
        test_print_interval = 20
        test_predictions = []
        for n_epi in range(1):
            done = False
            s = self.env.reset(all=False, test=True)
            if s == 'Sensemaking':
                s = [1, 0, 0]
            elif s == 'Foraging':
                s = [0, 1, 0]
            else:
                s = [0, 0, 1]
            s = np.array(s)
            predictions = []
            actions = []
            score=0
            while not done:
                for t in range(self.n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = self.env.step(s, a, True)
                    predictions.append(info)
                    ######################

                    if s_prime == 'Sensemaking':
                        s_prime = [1, 0, 0]
                    elif s_prime == 'Foraging':
                        s_prime = [0, 1, 0]
                    else:
                        s_prime = [0, 0, 1]
                    s_prime = np.array(s_prime)

                    ################

                    ################
                    model.put_data((s, a, r, s_prime, done))
                    s = s_prime
                    actions.append(a)

                    score += r

                    if done:
                        break
                model.train_net()

            print("#Test of episode :{}, avg score : {:.1f}, accuracy: {:.1f}, actions: {}".format(n_epi,
                                                                                                   score,
                                                                                                   np.mean(predictions),
                                                                                                   Counter(actions)))
            test_predictions.append(np.mean(predictions))
            print("############ Test Accuracy :{},".format(np.mean(predictions)))
        return np.mean(test_predictions)




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

    dataframe_users=[]
    dataframe_threshold = []
    dataframe_learningrate = []
    dataframe_discount = []



    aggregate_plotter = plotting.plotter(None)
    y_accu_all = []
    miscelleneous = misc.misc([])
    learning_rates=[0.0001,0.001,0.01, 0.1, 0.5,1]
    gammas=[0.05, 0.1, 0.2,0.5, 0.8, 0.9,0.99]

    for u in user_list:
        threshold_h = get_threshold(e, u)
        plotter = plotting.plotter(threshold_h)
        y_accu = []
        user_name=  miscelleneous.get_user_name(u)

        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0
            best_agent =None
            best_model =None
            for learning_rate in learning_rates:
                for gamma in gammas:
                    env = environment2.environment2()
                    env.process_data(u, thres)
                    agent = Agent(env,learning_rate,gamma)
                    model,accuracies = agent.train()


                    if accuracies > max_accu:
                        max_accu=accuracies
                        best_learning_rate=learning_rate
                        best_gamma=gamma
                        best_agent=agent
                        best_model=model
            print(
                "#TRAINING: User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}".format(user_name, thres, max_accu,
                                                                                            best_learning_rate,
                                                                                            best_gamma))
            test_accuracy=best_agent.test(best_model)

            y_accu.append(test_accuracy)
            dataframe_users.append(user_name)
            dataframe_threshold.append(thres)
            dataframe_learningrate.append(best_learning_rate)
            dataframe_discount.append(best_gamma)
            # dataframe_rollout.append(rollout)
            print("#TESTING User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}".format(user_name, thres, max_accu,best_learning_rate,best_gamma))
        plotter.plot_main(y_accu, user_name)
        y_accu_all.append(y_accu)
    title= "actor-critic" + exp
    aggregate_plotter.aggregate(y_accu_all, title)
    result_dataframe['User']= dataframe_users
    result_dataframe['Threshold']= dataframe_threshold
    result_dataframe['LearningRate'] = dataframe_learningrate
    result_dataframe['Discount'] = dataframe_discount
    result_dataframe.to_csv("data/NDSI-2D\\"+title+".csv", index=False)


def get_user_name(url):
        string = url.split('\\')
        fname = string[len(string) - 1]
        uname = ((fname.split('userid_')[1]).split('.'))[0]
        return uname



if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    miscelleneous= misc.misc([])
    user_list_experienced = np.array(
        ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
    user_list_first_time = np.setdiff1d(user_list_2D, user_list_experienced)

    p1 = multiprocessing.Process(target=user_set, args=(env,user_list_first_time[:6],'_first_time'))
    p2 = multiprocessing.Process(target=user_set, args=(env, user_list_experienced[:6],'_experienced'))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

