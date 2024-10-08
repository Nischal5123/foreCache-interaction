import numpy as np
from collections import defaultdict
import itertools
import environment_vizrec as environment_vizrec
import multiprocessing
import time
import random
from collections import Counter
import pandas as pd
import json
import concurrent.futures
import os
class TDLearning:
    def __init__(self,environment):
        self.env=environment
        self.user=None
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
                    best_action = random.randint(0, 3)
            else:
                best_action = np.argmax(Q[state])
            return best_action

        return policy_fnc



    def q_learning(self, user, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        Q = defaultdict(lambda: [0.0, 0.0, 0.0 , 0.0])
        self.user=user


        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))

            # Reset the environment and pick the first state
            state = self.env.reset()
            training_accuracy=[]
            for t in itertools.count():
                # Take a step
                action = policy(state)
                next_state, reward, done, info, ground_action = self.env.step(state, action, False)

                training_accuracy.append(info)

                best_next_action = np.argmax(Q[next_state])
                td_target = reward*info + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                # updating based on ground action
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][self.env.valid_actions.index(ground_action)]
                Q[state][self.env.valid_actions.index(ground_action)] += alpha * (td_delta)

                state = next_state
                if done:
                    break


        return Q, np.nanmean(training_accuracy)


    def test(self, Q, discount_factor, alpha, epsilon, threshold, num_episodes=1):
        epsilon = epsilon

        for i_episode in range(1):
            state = self.env.reset(all=False, test=True)
            stats = []
            insight = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(self.env.valid_actions))
            ground_truth = []
            all_predictions = []
            for t in itertools.count():
                # Take a step

                action = policy(state)
                next_state, reward, done, prediction,true_reward, ground_action, predicted_action, index = self.env.step(state, action, True)
                stats.append(prediction)
                insight[ground_action].append(prediction)


                # print(prediction)
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward*prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                # updating based on ground action
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][self.env.valid_actions.index(ground_action)]
                Q[state][self.env.valid_actions.index(ground_action)] += alpha * (td_delta)

                state = next_state
                ground_truth.append(ground_action)
                all_predictions.append(predicted_action)

                if done:
                    break
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.nanmean(stats), granular_prediction,  all_predictions, ground_truth
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

    y_true_all = []
    y_pred_all = []
    best_granularPredictions= {}

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
                        obj = TDLearning(env)
                        Q, train_accuracy = obj.q_learning(get_user_name(u), 50, gamma, learning_rate, eps)
                        if train_accuracy > max_accu:
                            max_accu = train_accuracy
                            best_learning_rate = learning_rate
                            best_gamma = gamma
                            best_agent = obj
                            best_model = Q
                            best_eps=eps
        print("#TRAINING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}".format(
                        user_name, thres, max_accu, best_learning_rate, best_gamma, best_eps))

        #best agent automatically gets all the threshold, Q information since its a full Object Store
        best_test_accs=0
        best_y_pred=[]
        best_y_true=[]
        # test_accs=[]
        for i in range(5):
            test_model=best_model
            test_agent=best_agent
            test_accuracy, granularPredictions, all_predictions, ground_truth = test_agent.test(test_model, best_gamma, best_learning_rate,
                                                                     best_eps, thres, 1)
            if test_accuracy>best_test_accs:
                best_test_accs=test_accuracy
                best_y_pred=all_predictions
                best_y_true=ground_truth
                best_granularPredictions=granularPredictions

        test_accuracy=best_test_accs
        # test_accuracy=np.mean(test_accs)

        y_accu.append(test_accuracy)
        accuracy_per_state = 0
        result_dataframe_user = pd.concat([result_dataframe_user, pd.DataFrame({
            'User': [user_name],
            'Threshold': [thres],
            'LearningRate': [best_learning_rate],
            'Discount': [best_gamma],
            'Accuracy': [test_accuracy],
            'GranularPredictions': [str(best_granularPredictions)],
            'Predictions': [str(best_y_pred)],
            'GroundTruth': [str(best_y_true)],
            'Algorithm': [algo],
            'Temperature': [0],
            'Epsilon':[best_eps]
        })], ignore_index=True)

        print(
            "#TESTING: User: {}, Threshold: {:.1f}, Accuracy: {}, LR: {}, Discount: {}, Epsilon: {}, Split_Accuracy: {}".format(
                user_name, thres, test_accuracy, best_learning_rate, best_gamma, best_eps, accuracy_per_state))
    return result_dataframe_user, y_accu, y_true_all, y_pred_all

def run_experiment(user_list, algo, hyperparam_file,dataset,task):
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    result_dataframe= pd.DataFrame(
        columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                 'StateAccuracy', 'Reward', 'Epsilon'])


    title = algo


    y_accu_all = []
    y_true_all = []
    y_pred_all = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment_for_user, u, algo, hyperparams): u for u in user_list}

        for future in concurrent.futures.as_completed(futures):
            user_result_dataframe, user_y_accu, user_y_true, user_y_pred = future.result()
            result_dataframe = pd.concat([result_dataframe, user_result_dataframe], ignore_index=True)
            y_accu_all.append(user_y_accu)
            y_true_all.extend(user_y_true)
            y_pred_all.extend(user_y_pred)

    # Define the output directory and file name
    directory = f"Experiments_Folder/Individual-Model/{dataset}/{task}"
    os.makedirs(directory, exist_ok=True)
    output_file = f"{directory}/QLearnAligned-Individual-Model.csv"

    result_dataframe.to_csv(output_file, index=False)




if __name__ == '__main__':
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']
    overall_accuracy = []
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = sorted(env.get_user_list(dataset, task))
            run_experiment(user_list_name, 'QLearn', 'sampled-hyperparameters-config.json', dataset, task)
            #read csv and get the accuracy
            df = pd.read_csv(f"Experiments_Folder/Individual-Model/{dataset}/{task}/QLearnAligned-Individual-Model.csv")
            overall_accuracy.append(np.mean(df['Accuracy']))


    print(np.mean(overall_accuracy))

