import numpy as np
from collections import defaultdict
import itertools
import environment2 as environment2
import time
import random
import plotting

learning_rate = 0.0002
gamma = 0.99

class Qlearning:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.eps=0.9


    def select_action(self, s):
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 1)
        else:
            action_val = self.q_table[s]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        a_prime = self.select_action(s_prime)  #
        self.q_table[s][a] += learning_rate * (r + (gamma * np.argmax(self.q_table[s_prime])) - self.q_table[s][a])

    def anneal_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        print(self.q_table.items())



def Q_train(env):
    agent = Qlearning()

    for n_epi in range(1000):
        done = False
        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done, info = env.step(s,a,False)
            agent.update_table((s, a, r, s_prime))
            s = s_prime
        agent.anneal_eps()

    agent.show_table()
    return agent

def Q_test(agent,env) :
    test_agent=agent
    test_accuracies=[]
    for n_epi in range(100):
        done = False
        s = env.reset(all=False , test=True)
        predictions=[]
        while not done:
            a = test_agent.select_action(s)
            s_prime, r, done, info = env.step(s,a,True)
            test_agent.update_table((s, a, r, s_prime))
            s = s_prime
            predictions.append(info)
        test_agent.anneal_eps()

        print("# of episode :{}, accuracy : {}".format(n_epi, np.mean(predictions)))
        test_accuracies.append(np.mean(predictions))
    test_agent.show_table()

    return test_accuracies

def main():
    env = environment2.environment2()
    users = env.user_list_2D
    env.process_data(users[0], 0.8)
    agent=Q_train(env)
    accuracies=Q_test(agent,env)
    print(accuracies)


if __name__ == '__main__':
        main()