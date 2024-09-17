import numpy as np
from collections import defaultdict
import itertools
import environment_vizrec as environment5
import concurrent.futures
import os
import json
import random


class SARSA:
    def __init__(self):
        pass

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fnc(state):
            coin = random.random()
            if coin < epsilon:
                best_action = random.randint(0, nA - 1)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc

    def sarsa_learning(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        for i_episode in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
            state = env.reset()
            training_accuracy = []
            action = policy(state)
            for t in itertools.count():

                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                training_accuracy.append(prediction)
                next_action = policy(next_state)
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                action = next_action
                state = next_state
                if done:
                    break
        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon
        stats = []
        state = env.reset(all=False, test=True)
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        insight = defaultdict(list)
        action = policy(state)
        for _ in range(1):
            state = env.reset()
            for t in itertools.count():

                next_state, reward, done, prediction, _, ground_action, _, _ = env.step(state, action, True)
                stats.append(prediction)
                insight[ground_action].append(prediction)
                next_action = policy(next_state)
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                action = next_action
                state = next_state
                if done:
                    break

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        return np.mean(stats), granular_prediction


def training(train_files, env, algorithm, epoch):
    hyperparam_file = 'sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)
    discount_h = hyperparams['gammas']
    alpha_h = hyperparams['learning_rates']
    epsilon_h = hyperparams['epsilon']

    best_discount = best_alpha = best_eps = max_accu = -1
    for eps in epsilon_h:
        for alp in alpha_h:
            for dis in discount_h:
                accu = []
                model = SARSA()
                Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))
                for user in train_files:
                    env.reset(True)
                    env.process_data(user, 0)
                    Q, accu_user = model.sarsa_learning(Q, env, epoch, dis, alp, eps)
                    accu.append(accu_user)

                accu_model = np.mean(accu)
                if accu_model > max_accu:
                    max_accu = accu_model
                    best_eps = eps
                    best_alpha = alp
                    best_discount = dis
                    best_q = Q
    return best_q, best_alpha, best_eps, best_discount


def testing(test_files, env, trained_Q, alpha, eps, discount, algorithm):
    Q = trained_Q
    final_accu = []
    for user in [test_files]:
        env.reset(True)
        env.process_data(user, 0)
        model = SARSA()
        accu, granular_acc = model.test(env, Q, discount, alpha, eps)
        final_accu.append(accu)
    return np.mean(final_accu), granular_acc


def process_user(user_data):
    #print(f"Processing Test User: {user_data[0]}")

    test_user, train_users, env, algorithm, epoch = user_data
    trained_Q, best_alpha, best_eps, best_discount = training(train_users, env, algorithm, epoch)

    best_accu = -1
    best_granular_acc = {}
    for _ in range(5):
        accu, granular_acc = testing(test_user, env, trained_Q, best_alpha, best_eps, best_discount, algorithm)
        if accu > best_accu:
            best_accu = accu
            best_granular_acc = granular_acc

    return [test_user, best_accu, best_granular_acc]


if __name__ == "__main__":
    env = environment5.environment_vizrec()
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            final_output = []
            print("# ", d, " Dataset", task, " Task")

            user_list = list(env.get_user_list(d, task))

            accuracies = []
            # Assuming you need to pass dataset ('d') and task ('task') along with the other parameters
            user_data_list = [(user_list[i], user_list[:i] + user_list[i + 1:], env, 'SARSA', 50)
                              for i in range(len(user_list))]

            # Using concurrent futures for parallel processing of users
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_user, user_data) for user_data in user_data_list]
                for future in concurrent.futures.as_completed(futures):
                    test_user, best_accu, best_granular_acc = future.result()
                    accuracies.append(best_accu)
                    final_output.append([test_user, best_accu, best_granular_acc])

            final_output = np.array(final_output)
            directory = f"Experiments_Folder/VizRec/{d}/{task}"
            os.makedirs(directory, exist_ok=True)
            np.savetxt(f"{directory}/SARSA-Single-Model.csv", final_output, delimiter=',', fmt='%s')

            accu = np.mean(accuracies)
            print("Dataset: {}, Task: {}, SARSA, {}".format(d, task, accu))
            overall_accuracy.append(accu)

    print("Overall Accuracy: {}".format(np.mean(overall_accuracy)))
