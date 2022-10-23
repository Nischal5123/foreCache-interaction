import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sys
import plotting
import environment2Location as environment 2
from tqdm import tqdm
import time
import multiprocessing

class TD_SARSA:
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
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc
    
    def sarsa(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        """
        SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
        
        Args:
            env: OpenAI environment.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance the sample a random action. Float betwen 0 and 1.
        
        Returns:
            A tuple (Q, stats).
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
        
        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            # if (i_episode + 1) % 100 == 0:
            #     print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            #     sys.stdout.flush()
            
            # Reset the environment and pick the first action
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # One step in the environment
            for t in itertools.count():
                # Take a step
                next_state, reward, done, _ = env.step(state, action, False)
                
                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                
                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
        
                if done:
                    break
                    
                action = next_action
                state = next_state        
        
        return Q, stats

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

            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

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
    users_b = env.user_list_bright
    users_f = env.user_list_faa

    obj2 = misc.misc(len(users_b))
    # best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_b, 'sarsa', 1)
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env, users_b[:4], 'sarsa', 5, ))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env, users_b[4:], 'sarsa', 5, ))


    obj2 = misc.misc(len(users_f))
    # best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_f, 'sarsa', 1)
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env, users_f[:4], 'sarsa', 5,))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env, users_f[4:], 'sarsa', 5,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    interval = round(time.time() - start_time)
    # pdb.set_trace()
    # print(interval)
    print("Executed time: {}:{}:{} ".format(int(interval / 3600), int((interval % 3600) / 60), interval % 60))
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
#     thres = 0.75
#     obj2 = misc.misc(len(users_hyper))
#     #hyper-param training
#     best_eps, best_discount, best_alpha = obj2.hyper_param(env, users_hyper, 'sarsa', 30)
#     #testing the model
#     # obj2.run_stuff(env, users_f, 20, 'SARSA_faa', best_eps, best_discount, best_alpha, 'sarsa')
#     # obj2.run_stuff(env, users_b, 20, 'SARSA_brightkite', best_eps, best_discount, best_alpha, 'sarsa')