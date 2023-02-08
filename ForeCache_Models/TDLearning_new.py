import misc_new as misc
import numpy as np
from collections import defaultdict
import itertools
import environment2 as environment2
import multiprocessing
import time
import random


class TDLearning:
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
                best_action = random.randint(0, 1)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc

    def q_learning(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5,step_size=0.01,decay_rate=0.001):
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
        #Q = defaultdict(lambda:  np.random.uniform(low=0, high=0.3,size=len(env.valid_actions)))
        Q = defaultdict(lambda: np.zeros(len(env.valid_actions)))

        stats = None
        step_size = step_size

        for i_episode in range(num_episodes):


            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            if epsilon > step_size:
                epsilon = max(epsilon - step_size, 0)

            # Reset the environment and pick the first state
            state = env.reset()
            # One step in the environment
            training_accuracy = []
            for t in itertools.count():
                # Take a step
                action = policy(state)

                next_state, reward, done, info = env.step(state, action, False)
                training_accuracy.append(info)

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                if done:
                    break
                state = next_state
        return Q,  np.mean(training_accuracy) #treunr training accuracy from last episode

    # @jit(target ="cuda")
    def test(self, env, Q, discount_factor, alpha, epsilon,num_episodes=1,step_size=0.01):
        epsilon = epsilon



        for i_episode in range(1):
            # Reset the environment and pick the first action
            state = env.reset(all=False, test=True)
            stats = []
            model_actions=[]
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            # One step in the environment
            for t in itertools.count():
                # Take a step
                action = policy(state)
                model_actions.append(action)
                next_state, reward, done, prediction = env.step(state, action, True)

                stats.append(prediction)

                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                if done:
                    break
                if epsilon > step_size:
                    epsilon = max(epsilon - step_size, 0)

                state = next_state

            cnt = 0
            for i in stats:
                cnt += i
            cnt /= len(stats)
        return cnt, model_actions


if __name__ == "__main__":
    start_time = time.time()
    env = environment2.environment2()
    user_list_2D = env.user_list_2D

    user_list_experienced=np.array(['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
                                    #'data/NDSI-2D\\taskname_ndsi-2d-task_userid_3abeecbe-327a-441e-be2a-0dd3763c1d45.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
                                    'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
                                    # only training users
                                    # user_list_2D = ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_44968286-f204-4ad6-a9b5-d95b38e97866.csv',
                                    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_3abeecbe-327a-441e-be2a-0dd3763c1d45.csv',
                                    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
                                    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_72a8d170-77ae-400e-b2a5-de9e1d33a714.csv',
                                    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_733a1ac5-0b01-485e-9b29-ac33932aa240.csv']
    user_list_first_time=np.setdiff1d(user_list_2D, user_list_experienced)
    user_list_3D = env.user_list_3D

    obj2 = misc.misc(len(user_list_2D))
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list_experienced[:6], 'qlearning',50,))
    p3 = multiprocessing.Process(target=obj2.hyper_param,args=(env, user_list_first_time[:6], 'qlearning', 50,))
    #
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_experienced[:6], 'sarsa', 50,))
    p4 = multiprocessing.Process(target=obj2.hyper_param,args=(env, user_list_first_time[:6], 'sarsa', 50,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
