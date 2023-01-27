import numpy as np
from collections import defaultdict
import itertools
import environment2 as environment2
import multiprocessing
import time


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

        # @jit(target ="cuda")
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    # @jit(target ="cuda")
    def sarsa(self, user, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5,step_size=0.01,decay_rate=0.001):
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
        Q = defaultdict(lambda:  np.random.uniform(low=0, high=0.3,size=len(env.valid_actions)))
        stats = None
        step_size = step_size

        # for i_episode in tqdm(range(num_episodes)):
        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            if epsilon > step_size:
                epsilon = max(epsilon/(1+i_episode*decay_rate),0)
            # Reset the environment and pick the first state
            state = env.reset()
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            for t in itertools.count():
                # Take a step

                next_state, reward, done, _ = env.step(state, action, False)

                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)



                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                if done:
                    break
                action= next_action
                state = next_state

        return Q, stats

    def test(self, env, Q, discount_factor, alpha,epsilon,num_episodes=10,step_size=0.01):
        epsilon = epsilon
        epsilon = 0.1
        for i_episode in range(num_episodes):

            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
            # Reset the environment and pick the first action
            state = env.reset(all=False, test=True)

            stats = []

            # One step in the environment
            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction = env.step(state, action, True)
                stats.append(action)

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

            if epsilon > step_size:
                epsilon = max(epsilon - step_size, 0)

            cnt = 0
            for i in stats:
                cnt += i
            cnt /= len(stats)

        return cnt,stats


if __name__ == "__main__":
    start_time = time.time()
    env = environment2.environment2()
    user_list_2D = env.user_list_2D

    user_list_experienced = np.array(
                                    ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
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
    user_list_first_time = np.setdiff1d(user_list_2D, user_list_experienced)
    user_list_3D = env.user_list_3D
    obj2 = misc.misc(len(user_list_2D))
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_experienced[:4], 'sarsa', 5,))
    p2 = multiprocessing.Process(target=obj2.hyper_param,
                                 args=(env, user_list_first_time[:4], 'sarsa', 5,))
    p1.start()

    p2.start()


    p1.join()

    p2.join()

