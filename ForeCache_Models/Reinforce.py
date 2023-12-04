import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment2
import plotting
from collections import Counter,defaultdict
import json
import concurrent.futures
eps=1e-35
class Policy(nn.Module):
    def __init__(self,learning_rate,gamma,tau):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 3)
        self.gamma=gamma
        self.temperature = tau
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x / self.temperature
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
    def __init__(self,env,learning_rate,gamma,tau):
        self.env = env
        self.learning_rate, self.gamma, self.temperature = learning_rate, gamma, tau
        self.pi = Policy(self.learning_rate, self.gamma,self.temperature)
        self.state_encoding = {
            "Sensemaking": [1, 0, 0],
            "Foraging": [0, 1, 0],
            "Navigation": [0, 0, 1]
        }

    def convert_idx_state(self, state_idx):
        state = next((key for key, value in self.state_encoding.items() if np.array_equal(value, state_idx)), None)
        return state

    def convert_state_idx(self, state):
        state_idx = self.state_encoding[state]
        return state_idx

    def train(self):
        score=0.0
        print_interval=50
        all_predictions=[]
        for n_epi in range(50):
            s = self.env.reset()
            s=np.array(self.convert_state_idx(s))
            done = False
            actions =[]
            predictions=[]
            while not done:
                prob = self.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info,_ = self.env.step(self.convert_idx_state(s),a,False)
                predictions.append(info)


                self.pi.put_data((r, prob[a]))

                s_prime = np.array(self.convert_state_idx(s_prime))
                s = s_prime

                score += r

            self.pi.train_net()
            all_predictions.append(np.mean(predictions))
        print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
        return self.pi, (np.mean(predictions)) #return last train_accuracy


    def test(self,policy):
        test_accuracies = []
        split_accuracy = defaultdict(list)

        for n_epi in range(1):
            s = self.env.reset(all=False, test=True)
            s = np.array(self.convert_state_idx(s))
            done = False
            predictions = []
            actions = []

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info,true_reward = self.env.step(self.convert_idx_state(s), a, True)
                predictions.append(info)
                split_accuracy[self.convert_idx_state(s)].append(info)


                policy.put_data((r, prob[a]))

                s_prime = np.array(self.convert_state_idx(s_prime))
                s = s_prime



                self.pi.train_net()

            print("TEST: # of episode :{}, accuracy : {}, actions: {}".format(n_epi, np.mean(predictions),
                                                                              Counter(actions)))

            test_accuracies.append(np.mean(predictions))
        return np.mean(test_accuracies),split_accuracy,0


def get_threshold(env, user):
    env.process_data(user, 0)
    counts = Counter(env.mem_roi)
    proportions = []
    total_count = len(env.mem_roi)

    for i in range(1, max(counts.keys()) + 1):
        current_count = sum(counts[key] for key in range(1, i + 1))
        proportions.append(current_count / total_count)
    return proportions[:-1]


def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname


def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state


def run_experiment(user_list,algo,hyperparam_file):
    # Load hyperparameters from JSON file
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward'])

    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
    temperatures = hyperparams['temperatures']

    aggregate_plotter = plotting.plotter(None)
    y_accu_all = []

    for u in user_list:
        # Extract user-specific threshold values
        threshold_h = hyperparams['threshold']
        plotter = plotting.plotter(threshold_h)
        y_accu = []
        user_name = get_user_name(u)
        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0
            best_agent=None
            best_policy=None
            best_temp=0

            for learning_rate in learning_rates:
                for gamma in gammas:
                    for temp in temperatures:
                        env = environment2.environment2()
                        env.process_data(u, thres)
                        agent = Reinforce(env,learning_rate,gamma,temp)
                        policy,accuracies = agent.train()

                        if accuracies > max_accu:
                            max_accu=accuracies
                            best_learning_rate=learning_rate
                            best_gamma=gamma
                            best_agent = agent
                            best_policy = policy
                            best_temp=temp

            print("#TRAINING: User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature:{}".format(user_name, thres,
                                                                                                     max_accu,
                                                                                                     best_learning_rate,
                                                                                                     best_gamma,best_temp))
            test_accuracy, split_accuracy,reward = best_agent.test(best_policy)
            accuracy_per_state = format_split_accuracy(split_accuracy)
            y_accu.append(test_accuracy)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [best_learning_rate],
                'Discount': [best_gamma],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [accuracy_per_state],
                'Algorithm': [algo],
                'Reward': [reward]
            })], ignore_index=True)
            print("#TESTING User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature: {}".format(user_name, thres,
                                                                                                     max_accu,
                                                                                                     best_learning_rate,
                                                                                                     best_gamma,best_temp))
        plotter.plot_main(y_accu, user_name)
        y_accu_all.append(y_accu)
    title = algo

    result_dataframe.to_csv("Experiments_Folder\\" + title + ".csv", index=False)

if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D[:3]
    run_experiment(user_list_2D, 'Reinforce', 'sampled-hyperparameters-config.json')


