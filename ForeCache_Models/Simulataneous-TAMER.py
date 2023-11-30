import environment2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import pandas as pd
from collections import defaultdict

# Neural network architecture for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QLearningAgent:
    def __init__(self, env, learning_rate, gamma, epsilon, num_rollouts=10):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_rollouts = num_rollouts
        self.state_dim = 3
        self.action_dim = 3
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Map state strings to numerical indices
        self.state_encoding = {
            "Sensemaking": [1, 0, 0],
            "Foraging": [0, 1, 0],
            "Navigation": [0, 0, 1]
        }

        # Initialize eligibility traces parameters
        self.cs = 1.0  # Scaling parameter for eligibility traces influence
        self.a = 0.1  # Accumulation factor for eligibility traces
        self.eligibility_traces = torch.zeros(self.state_dim)

    def convert_state_idx(self, state):
        state_idx = self.state_encoding[state]
        return state_idx

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state_idx = self.convert_state_idx(state)
            state_tensor = torch.FloatTensor(state_idx)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train(self):
        all_predictions = []
        for n_epi in range(50):
            done = False
            s = self.env.reset()
            predictions = []
            while not done:
                for t in range(self.num_rollouts):
                    a = self.epsilon_greedy(s)
                    s_prime, r, done, info, _ = self.env.step(s, a, False)
                    predictions.append(info)

                    #HANDLE STATES--> TENSOR
                    # Compute Q-learning target using eligibility traces
                    state_idx = self.convert_state_idx(s)
                    next_state_idx = self.convert_state_idx(s_prime)
                    state_tensor = torch.FloatTensor(state_idx)
                    next_state_tensor = torch.FloatTensor(next_state_idx)

                    # Compute shaped reward and eligibility traces influence (β)
                    shaped_reward = r + self.eligibility_traces.dot(state_tensor)
                    eligibility_influence = self.eligibility_traces.dot(state_tensor)
                    beta = self.cs * eligibility_influence


                    q_values = self.q_network(state_tensor)
                    next_q_values = self.q_network(next_state_tensor)
                    td_target = shaped_reward + self.gamma * next_q_values.max().item() * beta

                    # Calculate the TD error
                    q_value_a = q_values[a]
                    td_error = td_target - q_value_a

                    # Update eligibility traces and Q-value
                    self.eligibility_traces = torch.min(torch.ones_like(self.eligibility_traces),
                                                        self.eligibility_traces + state_tensor * self.a)
                    q_values[a] += self.learning_rate * td_error * state_tensor

                    # Update the Q-network
                    self.optimizer.zero_grad()
                    q_value_a.backward()
                    self.optimizer.step()

                    s = s_prime

                    if done:
                        break

                all_predictions.append(np.mean(predictions))

        print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
        return np.mean(predictions), self.q_network

def run_experiment(user_list, algo, hyperparam_file):
    # Load hyperparameters from JSON file
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Epsilon', 'Accuracy',
                 'StateAccuracy', 'Reward'])
    title = algo
    # Extract hyperparameters from JSON file
    learning_rates = hyperparams['learning_rates']
    gammas = hyperparams['gammas']
    threshold_h = hyperparams['threshold']
    epsilons = hyperparams['epsilon']
    # Loop over all users
    for u in user_list:

        y_accu = []
        user_name = get_user_name(u)

        # Loop over all threshold values
        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0
            best_eps = 0
            best_agent = None
            best_model = None

            # Loop over all combinations of hyperparameters
            for epsilon in epsilons:
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        env = environment2.environment2()
                        env.process_data(u, thres)
                        agent = QLearningAgent(env, learning_rate, gamma, epsilon)
                        accuracies, model = agent.train()
                        # Keep track of best combination of hyperparameters
                        if accuracies > max_accu:
                            max_accu = accuracies
                            best_learning_rate = learning_rate
                            best_gamma = gamma
                            best_agent = agent
                            best_model = model
                            best_eps = epsilon

                    # Print training results
                print(
                    "#TRAINING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}".format(
                        user_name, thres, max_accu, best_learning_rate, best_gamma, best_eps))

            # Test the best agent and store results in DataFrame
            test_accuracy, split_accuracy = best_agent.test(best_model)

            accuracy_per_state = format_split_accuracy(split_accuracy)
            y_accu.append(test_accuracy)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [best_learning_rate],
                'Discount': [best_gamma],
                'Epsilon': [best_eps],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [accuracy_per_state],
                'Algorithm': [title],
                'Reward': [0]
            })], ignore_index=True)

    result_dataframe.to_csv("Experiments_Folder/Tamer_Q_Learning.csv", index=False)

def format_split_accuracy(accuracy_dict):
    main_states = ['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state = []
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None)  # no data for that state
    return accuracy_per_state

def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname

if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    run_experiment(user_list_2D, 'TAMER-Q', 'sampled-hyperparameters-config.json')

