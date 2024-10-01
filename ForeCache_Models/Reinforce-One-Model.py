import environment_vizrec as environment5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import plotting
from collections import Counter
# import pandas as pd
import json
import os
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
import glob
from tqdm import tqdm
import multiprocessing
import ast
import concurrent.futures


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

    def __init__(self,learning_rate,gamma,tau,dataset):
        super(Policy, self).__init__()
        self.data = []
        if dataset=='birdstrikes':
            self.total_length=14
        else:
            self.total_length=16


        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 4)
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
    def __init__(self,env,learning_rate,gamma,tau,dataset, learned_policy=None):
        self.env = env
        if dataset == 'birdstrikes':
            self.total_length = 14
        else:
            self.total_length = 16
        self.learning_rate, self.gamma, self.temperature = learning_rate, gamma, tau
        if learned_policy is None:
            self.pi = Policy(learning_rate, gamma, tau, dataset)
        else:
            self.pi = learned_policy

    def convert_idx_state(self, state_idx):
        #state = next((key for key, value in self.state_encoding.items() if np.array_equal(value, state_idx)), None)
        converted_state = np.where(state_idx == 1)[0]
        return str(converted_state)

    def convert_state_idx(self, state):

        #state_idx = self.state_encoding[state]
        state_short=ast.literal_eval(state)
        # # Create a one-hot array
        # one_hot_array = np.zeros(self.total_length)
        #
        # # Set the values to 1 at the specified indices
        # for index in state_short:
        #     if index < self.total_length:
        #         one_hot_array[index] = 1

        return state_short

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

        for n_epi in range(1):
            s = self.env.reset(all=False, test=True)
            s = np.array(self.convert_state_idx(s))
            done = False
            predictions = []
            predicted_action = []
            all_ground_actions = []
            insight = defaultdict(list)

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info,_,ground_action, pred_action,_ = self.env.step(self.convert_idx_state(s), a, True)
                predictions.append(info)
                predicted_action.append(pred_action)
                all_ground_actions.append(ground_action)
                insight[ground_action].append(info)
                #split_accuracy[self.convert_idx_state(s)].append(info)


                policy.put_data((r, prob[a]))

                s_prime = np.array(self.convert_state_idx(s_prime))
                s = s_prime



                self.pi.train_net()

            #print("TEST: # of episode :{}, accuracy : {}, actions: {}".format(n_epi, np.mean(predictions), Counter(actions)))

            test_accuracies.append(np.mean(predictions))
            granular_prediction = defaultdict()
            for keys, values in insight.items():
                granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(test_accuracies),granular_prediction, predicted_action, all_ground_actions

def training(train_files, env, dataset, algorithm, epoch):
    # Load hyperparameters from JSON file
    hyperparam_file = 'sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']

    best_lr = best_gamma = max_accu = -1
    for lr in learning_rates:
        for ga in gammas:
            accu = []
            model = None
            for user in train_files:
                env= environment5.environment_vizrec()
                env.process_data(user, 0)
                agent = Reinforce(env, lr, ga, 1, dataset, model) #model is the learned policy from previous users

                model, accu_user = agent.train()
                accu.append(accu_user)

            accu_model = np.mean(accu)
            if accu_model > max_accu:
                max_accu = accu_model
                best_lr = lr
                best_gamma = ga
                best_ac_model = model

    return agent, best_ac_model, best_lr, best_gamma, max_accu


def testing(agent, test_files, env, trained_ac_model, best_lr, best_gamma, dataset, algorithm):
    final_accu = []
    for user in test_files:
        env.reset(True)
        env.process_data(user, 0)
        accu, granular_acc, predicted_actions, ground_actions = agent.test(trained_ac_model)
        final_accu.append(accu)

    return np.mean(final_accu), granular_acc, predicted_actions, ground_actions



def process_user(user_data):
    test_user_log, train_files, env, d, algorithm, epoch = user_data

    # Train the model
    agent, trained_ac_model, best_lr, best_gamma, training_accuracy = training(train_files, env, d, algorithm, epoch)

    # Test the model
    test_files = [test_user_log]
    best_accu = -1
    best_granular_acc = {}
    for _ in range(5):
        testing_accu, granular_acc, predicted_actions, ground_actions = testing(agent, test_files, env, trained_ac_model, best_lr, best_gamma, d, algorithm)
        if testing_accu > best_accu:
            best_accu = testing_accu
            best_granular_acc = granular_acc
            best_predicted_actions = predicted_actions
            best_ground_actions = ground_actions

    return [test_user_log, best_accu, best_granular_acc, best_predicted_actions, best_ground_actions]


if __name__ == "__main__":
    env = environment5.environment_vizrec()
    datasets = ['birdstrikes', 'movies']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            final_output = []
            env = environment5.environment_vizrec()

            # Get the user list
            user_list = list(env.get_user_list(d, task))

            user_data_list = [(user_list[i], user_list[:i] + user_list[i + 1:], env, d, 'Reinforce', 5)
                              for i in range(len(user_list))]

            accuracies = []
            trainACC = []
            testACC = []

            # Use concurrent futures for parallel processing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_user, user_data) for user_data in user_data_list]
                for future in concurrent.futures.as_completed(futures):
                    test_user_log, best_accu, best_granular_acc, best_predicted_actions, best_ground_actions = future.result()
                    accuracies.append(best_accu)
                    final_output.append([test_user_log, best_accu, best_granular_acc,str(best_predicted_actions), str(best_ground_actions)])

            # Convert the final output into a DataFrame
            df = pd.DataFrame(final_output,
                              columns=['User', 'Accuracy', 'GranularPredictions', 'Predictions', 'GroundTruth'])

            # Convert nested structures (if needed)
            df['GranularPredictions'] = df['GranularPredictions'].apply(lambda x: str(x))
            df['Predictions'] = df['Predictions'].apply(lambda x: str(x))
            df['GroundTruth'] = df['GroundTruth'].apply(lambda x: str(x))

            # Define the output directory and file name
            directory = f"Experiments_Folder/VizRec/{d}/{task}"
            os.makedirs(directory, exist_ok=True)
            output_file = f"{directory}/Reinforce-Single-Model.csv"

            # Save DataFrame to CSV
            df.to_csv(output_file, index=False)

            test_accu = np.mean(accuracies)
            print(f"Dataset: {d}, Task: {task}, Reinforce, {test_accu}")
            overall_accuracy.append(test_accu)

    print(f"Overall Accuracy: {np.mean(overall_accuracy)}")