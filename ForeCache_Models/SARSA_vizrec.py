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

class TD_SARSA:
    def __init__(self,environment):
        self.env = environment


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
                    best_action = random.randint(0, 4)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc

    def sarsa(
        self, user, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5
    ):
        """
               SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

               Args:
                   num_episodes: Number of episodes to run for.
                   discount_factor: Gamma discount factor.
                   alpha: TD learning rate.
                   epsilon: Chance the sample a random action. Float betwen 0 and 1.

               Returns:
                   A tuple (Q, stats).
                   Q is the optimal action-value function, a dictionary mapping state -> action values.
                   stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
               """


        Q = defaultdict(lambda: [0.0, 0.0, 0.0 , 0.0])

        stats = None

        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))

            # Reset the environment and pick the first state
            state = self.env.reset()
            action = policy(state)
            training_accuracy = []

            # print("episode")
            for t in itertools.count():
                # Take a step

                next_state, reward, done, info, _= self.env.step(state, action, False)

                next_action = policy(next_state)
                training_accuracy.append(info)

                # TD Update

                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                action = next_action
                state = next_state
                if done:
                    break

        return Q, np.mean(training_accuracy)

    def test(self, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for i_episode in range(1):


            # Reset the environment and pick the first action
            state = self.env.reset(all=False, test=True)

            stats = []
            reward_accumulated = [0.000000000000000000001]
            reward_possible = [0.000000000000000000001]
            split_accuracy = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))

            model_actions = []
            action = policy(state)
            for t in itertools.count():

                model_actions.append(action)
                next_state, reward, done, prediction,true_reward = self.env.step(state, action, True)
                stats.append(prediction)
                split_accuracy[state].append(prediction)
                reward_accumulated.append(reward)
                reward_possible.append(true_reward)

                # Pick the next action
                next_action = policy(next_state)

                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                action = next_action
                state = next_state
                if done:
                    break

        return np.mean(stats), model_actions, split_accuracy,np.mean(reward_accumulated)/np.mean(reward_possible)

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
    user_name = get_user_name(u)

    for thres in threshold_h:
        max_accu = -1
        best_learning_rate = 0
        best_gamma = 0
        best_eps = 0
        best_agent = None
        best_model = None

        for learning_rate in learning_rates:
            for gamma in gammas:
                for eps in epsilon:
                        env = environment_vizrec.environment_vizrec()
                        env.process_data(u, thres)
                        obj = TD_SARSA(env)
                        Q, train_accuracy = obj.sarsa(u, 50, gamma, learning_rate, eps)
                        if train_accuracy > max_accu:
                            max_accu = train_accuracy
                            best_learning_rate = learning_rate
                            best_gamma = gamma
                            best_agent = obj
                            best_model = Q
                            best_eps=eps
        print("#TRAINING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}".format(
                        user_name, thres, max_accu, best_learning_rate, best_gamma, best_eps))

        # best agent automatically gets all the threshold, Q information since its a full Object Store
        test_accs = []
        for i in range(5):
            test_model = best_model
            test_agent = best_agent
            test_accuracy, stats, split_accuracy, reward = test_agent.test(test_model, best_gamma, best_learning_rate,
                                                                           best_eps)
            test_accs.append(test_accuracy)
        test_accuracy = np.mean(test_accs)

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
            'Epsilon':[best_eps]
        })], ignore_index=True)

        print(
            "#TESTING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}, Split_Accuracy: {}".format(
                user_name, thres, test_accuracy, best_learning_rate, best_gamma, best_eps, accuracy_per_state))
    return result_dataframe_user, y_accu

def run_experiment(user_list, algo, hyperparam_file,task):
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    result_dataframe= pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward', 'Epsilon'])


    title = algo


    y_accu_all = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment_for_user, u, algo, hyperparams): u for u in user_list}

        for future in concurrent.futures.as_completed(futures):
            user_result_dataframe, user_y_accu = future.result()
            result_dataframe = pd.concat([result_dataframe, user_result_dataframe], ignore_index=True)
            y_accu_all.append(user_y_accu)


    result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}.csv".format(task,title), index=False)


if __name__ == '__main__':
    task = 'p4'
    env = environment_vizrec.environment_vizrec()
    user_list_2D = env.user_list_2D
    run_experiment(user_list_2D, 'SARSA', 'sampled-hyperparameters-config.json',task)

# if __name__ == "__main__":
#     start_time = time.time()
#     env = environment2.environment2()
#     user_list_2D = env.user_list_2D
#
#     obj2 = misc_vizrec.misc(len(user_list_2D))
#     p1 = multiprocessing.Process(
#         target=obj2.hyper_param, args=(env, user_list_2D, "SARSA", 50,)
#     )
#     p1.start()
#     p1.join()
