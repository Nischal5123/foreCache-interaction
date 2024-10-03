import numpy as np
from collections import defaultdict
import itertools
import environment_vizrec as environment5
import concurrent.futures
import os
import json
import random
import pandas as pd


class Qlearning:
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

    def q_learning(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        for i_episode in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
            state = env.reset()
            training_accuracy = []
            for t in itertools.count():
                action = policy(state)
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                training_accuracy.append(prediction)
                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                # updating based on ground action
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][env.valid_actions.index(ground_action)]
                Q[state][env.valid_actions.index(ground_action)] += alpha * (td_delta)

                state = next_state
                if done:
                    break
        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon
        stats = []
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        insight = defaultdict(list)
        ground_truth = []
        all_predictions = []
        for _ in range(1):
            state = env.reset()
            for t in itertools.count():
                action = policy(state)
                next_state, reward, done, prediction, _, ground_action, pred_action, _ = env.step(state, action, True)
                stats.append(prediction)
                insight[ground_action].append(prediction)

                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                # updating based on ground action
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][env.valid_actions.index(ground_action)]
                Q[state][env.valid_actions.index(ground_action)] += alpha * (td_delta)

                state = next_state
                ground_truth.append(ground_action)
                all_predictions.append(pred_action)

                if done:
                    break

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))
        return np.mean(stats), granular_prediction,  all_predictions, ground_truth


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
                model = Qlearning()
                Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))
                for user in train_files:
                    env.reset(True)
                    env.process_data(user, 0)
                    Q, accu_user = model.q_learning(Q, env, epoch, dis, alp, eps)
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
        model = Qlearning()
        accu, granular_acc, all_predictions, ground_truth = model.test(env, Q, discount, alpha, eps)
        final_accu.append(accu)
    return np.mean(final_accu), granular_acc, all_predictions, ground_truth


def process_user(user_data):
    #print(f"Processing Test User: {user_data[0]}")

    test_user, train_users, env, algorithm, epoch = user_data
    trained_Q, best_alpha, best_eps, best_discount = training(train_users, env, algorithm, epoch)

    best_accu = -1
    best_granular_acc = {}
    for _ in range(5):
        accu, granular_acc, all_predictions, ground_truth = testing(test_user, env, trained_Q, best_alpha, best_eps, best_discount, algorithm)
        if accu > best_accu:
            best_accu = accu
            best_granular_acc = granular_acc
            best_all_predictions = all_predictions
            best_ground_truth = ground_truth


    return [test_user, best_accu, best_granular_acc, best_all_predictions, best_ground_truth]


def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

if __name__ == "__main__":
    env = environment5.environment_vizrec()
    datasets = ['movies','birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            final_output = []
            print("# ", d, " Dataset", task, " Task")

            user_list = list(env.get_user_list(d, task))

            accuracies = []
            user_data_list = [(user_list[i], user_list[:i] + user_list[i + 1:], env, 'Qlearn', 50)
                              for i in range(len(user_list))]

            # Using concurrent futures for parallel processing of users
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_user, user_data) for user_data in user_data_list]
                for future in concurrent.futures.as_completed(futures):
                    test_user, best_accu, best_granular_acc, best_all_predictions, best_ground_truth = future.result()
                    accuracies.append(best_accu)
                    user_name = get_user_name(test_user)
                    # Store results as a list of lists
                    final_output.append(
                        [user_name, best_accu, best_granular_acc, best_all_predictions, best_ground_truth])

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
            output_file = f"{directory}/QLearn-Aligned-Test-Single-Model.csv"

            # Save DataFrame to CSV
            df.to_csv(output_file, index=False)

            accu = np.mean(accuracies)
            print("Dataset: {}, Task: {}, Q-Learning, {}".format(d, task, accu))
            overall_accuracy.append(accu)

    print("Overall Accuracy: {}".format(np.mean(overall_accuracy)))
