import pdb

import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sys
import plotting
import environment2
from tqdm import tqdm

class TDlearning:
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
        #
        # def policy_fnc(state):
        #     A = np.ones(nA, dtype=float) * epsilon / nA
        #     best_action = np.argmax(Q[state])
        #     A[best_action] += (1.0 - epsilon)
        #
        #     return A
        #
        # return policy_fnc

        def policy_fnc(state):
            prob_t = [0, 0]
            for a in range(nA):
                prob_t[a] = np.exp(Q[state][a] / 10)

            prob_t = np.true_divide(prob_t, sum(prob_t))
            print(prob_t)
            return prob_t
        return policy_fnc


    def q_learning(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
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

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

        for i_episode in tqdm(range(num_episodes)):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # Reset the environment and pick the first state
            state = env.reset()

            # One step in the environment
            # total_reward = 0.0
            # print("episode")
            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = env.step(state, action, False)
                # pdb.set_trace()
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                # pdb.set_trace()
                if done:
                    break
                state = next_state
        # print(policy)
        return Q, stats

    def test(self, env, Q, epsilon=0):

        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        discount_factor = 1.0
        alpha = 0.03
        # Reset the environment and pick the first action
        state = env.reset(all = False, test=True)
        valid_actions = ["same", "change"]
        stats = []
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, prediction = env.step(state, action, True)

            stats.append(prediction)
            # print(prediction)
            best_next_action = np.argmax(Q[next_state])
            print(valid_actions[best_next_action])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

        cnt = 0
        for i in stats:
            cnt += i
        cnt /= len(stats)
        print("Accuracy of Action Prediction: {}".format(cnt))
        # plt.imshow(Q, interpolation='none')
        # plt.show()
        return cnt

if __name__ == "__main__":
    accuracies = []
    plot_list=[]
    env = environment2.environment2()
    users = env.user_list
    for i in range(len(users)):
        plot_list.append(users[i][28:-10])
        epoch_user_accuracy=[]
        for epoch in range(1):
            #env = environment2.environment2()
            thres = 0.8 #the percent of interactions Q-Learning will be trained on
            print('########For user#############',users[i])
            env.process_data(users[i], thres)
            obj = TDlearning()
            Q, stats = obj.q_learning(users[i], env, 5000)
            epoch_user_accuracy.append(obj.test(env, Q)) #get test accuracy
            env.reset(True, False)
            # print("OK")
        accuracies.append(np.mean(epoch_user_accuracy))
    plt.plot(plot_list,accuracies, '-ro', label='Q learning Average Test Accuracy for Users 1-20')
    plt.xlabel("Users 1 - 20")
    plt.ylabel("Test Accuracy on action prediction")
    plt.legend(loc='upper left')
    plt.show()


#learn a policy a user will follow, convergance policy -> is too strict? Too much fluctuation
# MDP might not reflect -> how user's learn
#3rd possiblity -> humans follow different learning models in different setting.
# When humans face a problem with lot of actions -> model-free algorithms
# when environment is not complex -> model-based

# Completely open-ended workload
# Making sure uniform task design, correct design, relation between action and states.
# Criteria of convergence should not be too strict

# Difference between offline and online RL (Both, Relationship, with our work)
# ideally we would like to simulate what happened in the simulation when the user's participated in the system
# What we should do in response?
#


