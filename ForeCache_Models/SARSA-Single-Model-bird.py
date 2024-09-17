import numpy as np
from collections import defaultdict
import itertools
import environment_vizrec as environment5
import multiprocessing
from multiprocessing import Pool
import time
import random
from pathlib import Path
import glob
# from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import json
import pdb


class SARSA:
    def __init__(self):
        pass

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """

        def policy_fnc(state):
            coin = random.random()
            if coin < epsilon:
                best_action = random.randint(0, nA - 1)
                # print("random")
            else:
                best_action = np.argmax(Q[state])
                # print("best")
            return best_action

        return policy_fnc

    def sarsa(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        SARSA algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: setting the environment as local fnc by importing env earlier
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            # Reset the environment and pick the first state
            state = env.reset()
            action = policy(state)
            training_accuracy = []
            for t in itertools.count():
                # Take a step

                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                # print(Q[state], action, state)
                training_accuracy.append(prediction)
                next_action= policy(next_state)

                td_target = reward + discount_factor * Q[next_state][next_action]
                # print("State {}, action {}".format(state, action))
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                # print(Q[state], best_next_action, next_state, Q[next_state])
                # pdb.set_trace()

                # updating based on ground action
                # best_next_action = np.argmax(Q[next_state])
                # td_target = 1 + discount_factor * Q[next_state][best_next_action]
                # td_delta = td_target - Q[state][ground_action]
                # Q[state][ground_action] += alpha * (td_delta)
                action = next_action
                state = next_state
                if done:
                    break

        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):

            state = env.reset(all=False, test=True)
            stats = []

            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
            insight = defaultdict(list)
            action = policy(state)
            for t in itertools.count():


                # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                next_state, reward, done, prediction,_, ground_action,_,_ = env.step(state, action, True)
                # print(state, Q[state], ground_action, action, prediction)
                # if action == ground_action:
                #     reward = 10
                # else:
                #     reward = -10
                # pdb.set_trace()
                stats.append(prediction)
                insight[ground_action].append(prediction)

                # Pick the next action
                next_action = policy(next_state)
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                # updating based on ground action
                # best_next_action = np.argmax(Q[next_state])
                # td_target = 10 + discount_factor * Q[next_state][best_next_action]
                # td_delta = td_target - Q[state][ground_action]
                # Q[state][ground_action] += alpha * (td_delta)
                # print("Updated Q {}".format(Q[state]))
                action = next_action
                state = next_state
                if done:
                    break

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(stats), granular_prediction


def training(train_files, env, dataset, algorithm, epoch):
    # loading the hyper-parameters
    hyperparam_file = 'sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)
    # Extract hyperparameters from JSON file
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
                    # updates the Q value after each user trajectory
                    # print(user[0])
                    Q, accu_user = model.sarsa(Q, env, epoch, dis, alp, eps)
                    # print(user[0], eps, alp, dis, accu_user)
                    accu.append(accu_user)

                # accuracy of the model learned over training data
                accu_model = np.mean(accu)
                if accu_model > max_accu:
                    max_accu = accu_model
                    best_eps = eps
                    best_alpha = alp
                    best_discount = dis
                    best_q = Q
    #print("Training Accuracy", max_accu)
    #print(f"Best Hyperparameters Epsilon {best_eps}, Alpha {best_alpha}, Discount {best_discount}")
    return best_q, best_alpha, best_eps, best_discount


def testing(test_files, env, trained_Q, alpha, eps, discount, dataset, algorithm):
    Q = trained_Q
    final_accu = []
    for user in test_files:
        env.reset(True)
        env.process_data(user, 0)
        model = SARSA()
        accu, granular_acc = model.test(env, Q, discount, alpha, eps)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    # print("Q-Learning, {}, {:.2f}".format(k, np.mean(final_accu)))
    return np.mean(final_accu), granular_acc


if __name__ == "__main__":
    env = environment5.environment_vizrec()
    datasets = ['birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            final_output = []
            print("# ", d, " Dataset", task, " Task")

            user_list = list(env.get_user_list(d, task))

            accuracies = []
            # Implement LOOCV
            for i in range(len(user_list)):
                # Leave one user out for testing, the rest for training
                test_user = [user_list[i]]
                train_users = user_list[:i] + user_list[i+1:]

                # Train the model on the training users
                trained_Q, best_alpha, best_eps, best_discount = training(train_users, env, d, 'SARSA', 100)

                # Test the model on the single test user
                best_accu = -1
                best_granular_acc = {}
                for i in range(3):
                    accu, granular_acc = testing(test_user, env, trained_Q, best_alpha, best_eps, best_discount, d, 'SARSA')
                    if accu > best_accu:
                        best_accu = accu
                        best_granular_acc = granular_acc

               # print("Testing Accuracy for user {}: {:.2f}".format(test_user, accu))
                accuracies.append(best_accu)
                final_output.append([test_user[0], accu, best_granular_acc])
            final_output = np.array(final_output)
            directory = f"Experiments_Folder/VizRec/{d}/{task}"
            np.savetxt(f"{directory}/SARSA-Single-Model", final_output, delimiter=',', fmt='%s')

            # Calculate the mean accuracy across all LOOCV iterations
            accu = np.mean(accuracies)
            print("Dataset: {}, Task: {}, SARSA, {:.2f}".format(d, task, accu))
            overall_accuracy.append(accu)

    print("Overall Accuracy: {:.2f}".format(np.mean(overall_accuracy)))
