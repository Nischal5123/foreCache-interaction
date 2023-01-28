import environment2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import Counter

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
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
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main():

    env = environment2.environment2()
    users = env.user_list_2D

    for user in users[:5]:
        env = environment2.environment2()
        env.process_data(user, 0.8)
        model = ActorCritic()
        print_interval = 100
        score = 0.0

        for n_epi in range(1000):
            done = False
            s = env.reset()
            if s == 'Sensemaking':
                s = [1, 0, 0]
            elif s == 'Foraging':
                s = [0, 1, 0]
            else:
                s = [0, 0, 1]
            s=np.array(s)
            predictions=[]
            actions=[]
            while not done:
                for t in range(n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    actions.append(a)
                    s_prime, r, done, info = env.step(s,a,False)
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

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f}, accuracy: {:.1f} , actions{}".format(n_epi, score / print_interval,np.mean(predictions),Counter(actions)))
                score = 0.0

        for n_epi in range(100):
            done = False
            s = env.reset(all=False , test=True)
            if s == 'Sensemaking':
                s = [1, 0, 0]
            elif s == 'Foraging':
                s = [0, 1, 0]
            else:
                s = [0, 0, 1]
            s = np.array(s)
            predictions = []
            actions =[]
            while not done:
                for t in range(n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(s, a, True)
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



            print("# of episode :{}, avg score : {:.1f}, accuracy: {:.1f}, actions: {}".format(n_epi, score / print_interval,
                                                                                      np.mean(predictions),Counter(actions)))
            score = 0.0

if __name__ == '__main__':
    main()