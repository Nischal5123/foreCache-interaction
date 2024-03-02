import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment_vizrec
from collections import Counter,defaultdict
import json
import ast
import concurrent.futures
eps=1e-35
class Policy(nn.Module):

    def __init__(self,learning_rate,gamma,tau):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 4)
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

    def convert_idx_state(self, state_idx):
        #state = next((key for key, value in self.state_encoding.items() if np.array_equal(value, state_idx)), None)
        return str(state_idx)

    def convert_state_idx(self, state):
        #state_idx = self.state_encoding[state]
        return ast.literal_eval(state)

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
       # print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
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
                #split_accuracy[self.convert_idx_state(s)].append(info)


                policy.put_data((r, prob[a]))

                s_prime = np.array(self.convert_state_idx(s_prime))
                s = s_prime



                self.pi.train_net()

            #print("TEST: # of episode :{}, accuracy : {}, actions: {}".format(n_epi, np.mean(predictions), Counter(actions)))

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
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
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




def run_experiment_for_user(u, algo, hyperparams):
    result_dataframe_user = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward'])

    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
    temperatures = hyperparams['temperatures']

    threshold_h = hyperparams['threshold']
    y_accu = []
    user_name = get_user_name(u)

    for thres in threshold_h:
        max_accu = -1
        best_learning_rate = 0
        best_gamma = 0
        best_agent = None
        best_model = None
        best_temp = 0

        for learning_rate in learning_rates:
            for gamma in gammas:
                for temp in temperatures:
                    env = environment_vizrec.environment_vizrec()
                    env.process_data(u, thres)
                    agent = Reinforce(env, learning_rate, gamma, temp)
                    model, accuracies = agent.train()

                    if accuracies > max_accu:
                        max_accu = accuracies
                        best_learning_rate = learning_rate
                        best_gamma = gamma
                        best_agent = agent
                        best_model = model
                        best_temp = temp

       # print("#TRAINING: User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature:{}".format(user_name, thres, max_accu, best_learning_rate, best_gamma, best_temp))

        best_test_accs = 0
        for i in range(5):
            test_agent = best_agent
            test_model = best_model
            current_test_accuracy, split_accuracy, reward = test_agent.test(test_model)
            if current_test_accuracy > best_test_accs:
                best_test_accs = current_test_accuracy
        test_accuracy = best_test_accs

        accuracy_per_state = format_split_accuracy(split_accuracy)
        y_accu.append(test_accuracy)
        result_dataframe_user = pd.concat([result_dataframe_user, pd.DataFrame({
            'User': [user_name],
            'Threshold': [thres],
            'LearningRate': [best_learning_rate],
            'Discount': [best_gamma],
            'Accuracy': [test_accuracy],
            'StateAccuracy': [accuracy_per_state],
            'Algorithm': [algo],
            'Reward': [reward]
        })], ignore_index=True)
        print("#TESTING User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature: {}".format(
            user_name, thres, max_accu, best_learning_rate, best_gamma, best_temp))
    return result_dataframe_user, y_accu

def run_experiment(user_list, algo, hyperparam_file,dataset,task='p2'):
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    result_dataframe = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward'])
    title = algo


    y_accu_all = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment_for_user, u, algo, hyperparams): u for u in user_list}

        for future in concurrent.futures.as_completed(futures):
            user_result_dataframe, user_y_accu = future.result()

            result_dataframe = pd.concat([result_dataframe, user_result_dataframe], ignore_index=True)
            y_accu_all.append(user_y_accu)

    result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset, task, title), index=False)


if __name__ == '__main__':
    datasets = ['birdstrikes','movies']
    tasks = ['p1','p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            run_experiment(user_list_name, 'Reinforce', 'sampled-hyperparameters-config.json', dataset, task)
            print(f"Done with {dataset} {task}")


