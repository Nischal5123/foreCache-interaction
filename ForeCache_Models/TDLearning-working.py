import numpy as np
from collections import defaultdict
import itertools
import environment_vizrec
import multiprocessing
import time
import random
from collections import Counter
import pandas as pd
import json
import concurrent.futures
import matplotlib.pyplot as plt
import os


class TDLearning:
    def __init__(self, environment):
        self.env = environment
        self.user = None
        self.valid_actions = ['same', 'modify-1', 'modify-2', 'modify-3']

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fnc(state):
            coin = random.random()
            if coin < epsilon:
                best_action = random.randint(0, 3)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc

    def q_learning(self, user, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        self.user = user
        Q = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        for i_episode in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))
            state = self.env.reset()
            training_accuracy = []

            for t in itertools.count():
                action = policy(state)
                next_state, reward, done, info, _ = self.env.step(state, action, False)
                training_accuracy.append(info)
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                state = next_state
                if done:
                    break

        return Q, np.nanmean(training_accuracy)

    def test(self, user, Q, discount_factor, alpha, epsilon, thres, num_episodes=1):
        self.env=environment_vizrec.environment_vizrec()
        self.env.process_data(user, thres)
        epsilon = epsilon
        y_true = []
        y_pred = []
        stats = []
        split_accuracy = defaultdict(list)
        reward_accumulated = [0.000000000000000000001]
        reward_possible = [0.000000000000000000001]
        policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))

        length = len(self.env.mem_action)
        threshold = int(thres * length)

        for t in range(threshold, length):
            state = self.env.mem_states[t]
            action = policy(state)
            next_state = self.env.mem_states[t + 1] if t + 1 < len(self.env.mem_states) else state
            reward = self.env.mem_reward[t]
            true_action = self.env.mem_action[t]
            true_reward = reward  # Assuming true_reward is same as reward
            prediction = 1 if self.valid_actions[action] == true_action else 0

            y_true.append((true_action, t, self.user, thres))
            y_pred.append((self.valid_actions[action], t, self.user, thres))
            reward_accumulated.append(reward)
            split_accuracy[state].append(prediction)
            stats.append(prediction)
            reward_possible.append(true_reward)

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * (td_delta)


        return (np.nanmean(stats), y_pred, split_accuracy,
                np.nanmean(reward_accumulated) / np.nanmean(reward_possible), y_true)


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


def run_experiment_for_user(u, algo, hyperparams):
        result_dataframe_user = pd.DataFrame(
            columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                     'StateAccuracy', 'Reward', 'Epsilon'])
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        epsilon = hyperparams['epsilon']
        threshold_h = hyperparams['threshold']
        y_accu = []
        y_true_all = []
        y_pred_all = []
        user_name = get_user_name(u)

        for thres in threshold_h:
            max_accu = -1
            best_learning_rate = 0
            best_gamma = 0
            best_eps = 0
            best_agent = None
            best_model = None
            best_test_accs = 0
            best_y_pred = []
            best_y_true = []

            for learning_rate in learning_rates:
                for gamma in gammas:
                    for eps in epsilon:
                        env = environment_vizrec.environment_vizrec()
                        env.process_data(u, thres)
                        obj = TDLearning(env)
                        Q, train_accuracy = obj.q_learning(get_user_name(u), 50, gamma, learning_rate, eps)
                        if train_accuracy > max_accu:
                            max_accu = train_accuracy
                            best_learning_rate = learning_rate
                            best_gamma = gamma
                            best_agent = obj
                            best_model = Q
                            best_eps = eps


            # print(
            #     "#TRAINING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}".format(user_name,
            #                                                                                                      thres,
            #                                                                                                      max_accu,
            #                                                                                                      best_learning_rate,
            #                                                                                                      best_gamma,
            #                                                                                                      best_eps))


            for j in range(1):
                test_model = best_model
                test_agent = best_agent

                test_accuracy, y_pred, split_accuracy, reward, y_true = test_agent.test(u,test_model, best_gamma, best_learning_rate,
                                                                               best_eps,thres)
                if test_accuracy > best_test_accs:
                    best_test_accs = test_accuracy
                    best_y_pred = y_pred
                    best_y_true = y_true

            print("Saving Accuracy", thres)
            y_true_all.extend(best_y_true)
            y_pred_all.extend(best_y_pred)

            test_accuracy = best_test_accs
            y_accu.append(test_accuracy)
            accuracy_per_state = 0
            result_dataframe_user = pd.concat([result_dataframe_user, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [best_learning_rate],
                'Discount': [best_gamma],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [accuracy_per_state],
                'Algorithm': [algo],
                'Reward': [reward],
                'Temperature': [0],
                'Epsilon': [best_eps]
            })], ignore_index=True)

            print(
                "#TESTING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}, Split_Accuracy: {}".format(
                    user_name, thres, test_accuracy, best_learning_rate, best_gamma, best_eps, accuracy_per_state))

        return result_dataframe_user, y_accu, y_true_all, y_pred_all


def run_experiment(user_list, algo, hyperparam_file, dataset, task):
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    result_dataframe = pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward', 'Epsilon'])
    title = algo
    y_accu_all = []
    y_true_all = []
    y_pred_all = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment_for_user, u, algo, hyperparams): u for u in user_list}

        for future in concurrent.futures.as_completed(futures):
            user_result_dataframe, user_y_accu, user_y_true, user_y_pred = future.result()
            result_dataframe = pd.concat([result_dataframe, user_result_dataframe], ignore_index=True)
            y_accu_all.append(user_y_accu)
            y_true_all.extend(user_y_true)
            y_pred_all.extend(user_y_pred)

    result_dataframe.to_csv(f"Experiments_Folder/VizRec/{dataset}/{task}/{title}.csv", index=False)
    save_data_to_csv(y_true_all, y_pred_all, task, dataset)



def save_data_to_csv(y_true_all, y_pred_all, task, dataset, algorithm='QLearn'):
    data = []
    for yt, yp in zip(y_true_all, y_pred_all):
        data.append([yt[0], yt[1], yt[2], yp[0], yp[1], yp[3]])

    df = pd.DataFrame(data, columns=['y_true', 'interaction_point', 'user', 'y_pred', 'pred_interaction_point', 'threshold'])
    directory = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
    os.makedirs(directory, exist_ok=True)
    df.to_csv(f"{directory}/{algorithm}_predictions_data.csv", index=False)

if __name__ == '__main__':
    datasets = ['movies']
    tasks = ['p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = sorted(env.get_user_list(dataset, task))
            run_experiment(user_list_name, 'QLearn', 'sampled-hyperparameters-config.json', dataset, task)


