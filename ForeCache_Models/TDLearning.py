import misc
import numpy as np
from collections import defaultdict
import itertools
import environment2_changepoint as environment2
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
                if state == "Navigation":
                    best_action = random.randint(0, 2)
                else:
                    best_action = random.randint(0, 1)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc



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
        Q = {
            'Navigation':  np.zeros(3),
            'Foraging': np.zeros(2),
            'Sensemaking': np.zeros(2)
        }

        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            # Reset the environment and pick the first state
            state = env.reset()
            training_accuracy=[]
            for t in itertools.count():
                # Take a step
                action = policy(state)

                next_state, reward, done, info,_,cpd = env.step(state, action, False)

                training_accuracy.append(info)

                # TD Update
                if cpd:
                    Q = {
                        'Navigation': np.zeros(3),
                        'Foraging': np.zeros(2),
                        'Sensemaking': np.zeros(2)
                    }
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta + info)
                state = next_state
                if done:
                    break


        return Q, np.mean(training_accuracy)


    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for i_episode in range(1):

            state = env.reset(all=False, test=True)
            stats = []
            split_accuracy=defaultdict(list)

            model_actions = []
            reward_accumulated=[0.000000000000000000001]
            reward_possible=[0.000000000000000000001]
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            for t in itertools.count():
                # Take a step

                action = policy(state)
                model_actions.append(action)
                next_state, reward, done, prediction,true_reward,_ = env.step(state, action, True)
                reward_accumulated.append(reward)

                split_accuracy[state].append(prediction)
                stats.append(prediction)
                reward_accumulated.append(reward)
                reward_possible.append(true_reward)

                # print(prediction)
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta + prediction)

                state = next_state
                if done:
                    break

        return np.mean(stats),model_actions,split_accuracy, np.mean(reward_accumulated)/np.mean(reward_possible)


if __name__ == "__main__":
    start_time = time.time()
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    obj2 = misc.misc(len(user_list_2D))
    p8 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list_2D, 'QLearn',50,))
    #p10 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list_2D, 'SARSA',50,))
    p8.start()
    #p10.start()
    p8.join()
    #p10.join()


