import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
import sys
import plotting
import environment2Location as environment2
from tqdm import tqdm
# from numba import jit, cuda 
import multiprocessing
import time

class TDLearning:
    def __init__(self):
        pass

    # @jit(target ="cuda")
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
        
        # @jit(target ="cuda")
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    # @jit(target ="cuda")
    def q_learning(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
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
        # stats = plotting.EpisodeStats(
        #     episode_lengths=np.zeros(num_episodes),
        #     episode_rewards=np.zeros(num_episodes))
        stats = None
        # The policy we're following
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

        # for i_episode in tqdm(range(num_episodes)):
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            # if (i_episode + 1) % 100 == 0:
            #     print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            #     sys.stdout.flush()

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
                # stats.episode_rewards[i_episode] += reward
                # stats.episode_lengths[i_episode] = t

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

    # @jit(target ="cuda")
    def test(self, env, Q, discount_factor, alpha, epsilon):

        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
        # Reset the environment and pick the first action
        state = env.reset(all = False, test=True)

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
            # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions 
            # best_next_action = np.argmax(Q[next_state])
            # td_target = reward + discount_factor * Q[next_state][best_next_action]
            # td_delta = td_target - Q[state][action]
            # Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

        cnt = 0
        for i in stats:
            cnt += i
        cnt /= len(stats)
        # print("Accuracy of State Prediction: {}".format(cnt))
        return cnt


if __name__ == "__main__":
    start_time = time.time()
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    user_list_3D = env.user_list_3D

    obj2 = misc.misc(len(user_list_2D))
    # best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_b, 'sarsa', 1)
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_2D[:10], 'qlearning', 5,))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_2D[10:len(user_list_2D)], 'qlearning', 5,))

    obj2 = misc.misc(len(user_list_3D))
    # best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_f, 'sarsa', 1)
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_3D[:10], 'qlearning', 5,))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_3D[10:len(user_list_3D)], 'qlearning', 5,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

# if __name__ == "__main__":
#     env = environment2.environment2()
#     users_b = env.user_list_bright
#     users_f = env.user_list_faa
#     users_hyper = []
#     for i in range(8):
#         c = np.random.randint(0, len(users_b))
#         users_hyper.append(users_b[c])
#         users_b.remove(users_b[c])
#
#     for i in range(8):
#         c = np.random.randint(0, len(users_f))
#         users_hyper.append(users_f[c])
#         users_f.remove(users_f[c])
#
#     # thres = 0.75 #the percent of interactions Q-Learning will be trained on
#     obj2 = misc.misc(len(users_hyper))
#
#     # training hyper-parameters
#     best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_hyper, 'qlearning', 30)
#
#     # testing the model
#     # obj2.run_stuff(env, users_f, 30, 'QLearning_faa', best_eps, best_discount, best_alpha, 'qlearning')
#     # obj2.run_stuff(env, users_b, 30, 'QLearning_brightkite', best_eps, best_discount, best_alpha, 'qlearning')
#     # print(env.find_states)