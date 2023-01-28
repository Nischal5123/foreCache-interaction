import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment2
import numpy as np
import plotting
from collections import Counter

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 2)
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
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    env = environment2.environment2()
    users = env.user_list_2D
    env.process_data(users[0], 0.8)
    pi = Policy()
    score = 0.0
    print_interval = 20

    for n_epi in range(1000):
        s = env.reset()
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
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(s,a.item(),False)
            pi.put_data((r, prob[a]))
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

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}, actions :{}".format(n_epi, score / print_interval, Counter(actions)))
            score = 0.0


    test_accuracies=[]
    for n_epi in range(100):
        s = env.reset(all=False , test=True)
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
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(s,a.item(),True)
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





        print("# of episode :{}, accuracy : {}, actions: {}".format(n_epi, np.mean(predictions),Counter(actions)))
        score = 0.0
        test_accuracies.append(np.mean(predictions))
    return test_accuracies

if __name__ == '__main__':
    accuracies = main()
    plotting.plot_episode_stats(accuracies, 100, 'reinforce')