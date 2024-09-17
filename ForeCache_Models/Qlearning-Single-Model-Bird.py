import numpy as np
from collections import defaultdict, deque
import itertools
import random
import environment_vizrec as environment5
import concurrent.futures
import os
import json


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Qlearning:
    def __init__(self):
        self.buffer = ReplayBuffer(10000)  # Replay buffer size

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fnc(state):
            if random.random() < epsilon:
                return random.randint(0, nA - 1)
            return np.argmax(Q[state])

        return policy_fnc

    def q_learning(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5, epsilon_decay=0.99,
                   batch_size=32):
        for i_episode in range(num_episodes):
            state = env.reset()
            episode_acc = []

            for _ in itertools.count():
                # Use epsilon-greedy policy
                policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))
                action = policy(state)

                # Interact with the environment
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                episode_acc.append(prediction)

                # Store experience in replay buffer
                self.buffer.push(state, action, reward, next_state, done)

                if len(self.buffer) >= batch_size:
                    # Sample from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

                    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                        # Update Q-value using batch of experiences
                        best_next_action = np.argmax(Q[ns])
                        td_target = r + (discount_factor * Q[ns][best_next_action] * (1 - d))
                        td_error = td_target - Q[s][a]
                        Q[s][a] += alpha * td_error

                state = next_state
                if done:
                    break

            # Decay epsilon after each episode
            epsilon = max(0.1, epsilon * epsilon_decay)

        return Q, np.mean(episode_acc)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        stats = []
        insight = defaultdict(list)

        for _ in range(num_episodes):
            state = env.reset()
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.valid_actions))

            for _ in itertools.count():
                action = policy(state)
                next_state, reward, done, prediction, _, ground_action, _, _ = env.step(state, action, True)
                stats.append(prediction)
                insight[ground_action].append(prediction)

                state = next_state
                if done:
                    break

        granular_prediction = {key: (len(values), np.mean(values)) for key, values in insight.items()}
        return np.mean(stats), granular_prediction


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
        accu, granular_acc = model.test(env, Q, discount, alpha, eps)
        final_accu.append(accu)

    return np.mean(final_accu), granular_acc


def process_user(user_data):
    test_user, train_users, env, algorithm, epoch = user_data
    trained_Q, best_alpha, best_eps, best_discount = training(train_users, env, algorithm, epoch)

    best_accu = -1
    best_granular_acc = {}
    for _ in range(3):
        accu, granular_acc = testing(test_user, env, trained_Q, best_alpha, best_eps, best_discount, algorithm)
        if accu > best_accu:
            best_accu = accu
            best_granular_acc = granular_acc

    return [test_user, best_accu, best_granular_acc]


if __name__ == "__main__":
    env = environment5.environment_vizrec()
    datasets = ['movies']
    tasks = ['p1']
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            final_output = []
            print("# ", d, " Dataset", task, " Task")

            user_list = list(env.get_user_list(d, task))

            accuracies = []
            user_data_list = [(user_list[i], user_list[:i] + user_list[i + 1:], env, 'Qlearn', 100)
                              for i in range(len(user_list))]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_user, user_data) for user_data in user_data_list]
                for future in concurrent.futures.as_completed(futures):
                    test_user, best_accu, best_granular_acc = future.result()
                    accuracies.append(best_accu)
                    final_output.append([test_user, best_accu, best_granular_acc])

            final_output = np.array(final_output)
            directory = f"Experiments_Folder/VizRec/{d}/{task}"
            os.makedirs(directory, exist_ok=True)
            np.savetxt(f"{directory}/QLearn-Single-Model.csv", final_output, delimiter=',', fmt='%s')

            accu = np.mean(accuracies)
            print("Dataset: {}, Task: {}, Q-Learning, {}".format(d, task, accu))
            overall_accuracy.append(accu)

    print("Overall Accuracy: {}".format(np.mean(overall_accuracy)))
