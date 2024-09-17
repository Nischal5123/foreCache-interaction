import environment_vizrec as environment5
import numpy as np
import d3rlpy
import ast
import concurrent.futures
import csv

def process_user(test_user, user_list_name, env, d, task):
    """
    Process a single test user by training on the remaining users and evaluating the model.
    """
    print(f"Processing Test User: {test_user}")

    # Initialize training data containers
    train_states = []
    train_actions = []
    train_rewards = []
    train_terminals = []

    # Gather data from all other users for training
    for train_user in user_list_name:
        if train_user != test_user:  # Exclude the test user
            env.reset(True)
            env.process_data(train_user, 0)

            # Append the training user's data to the training set
            train_states.extend([ast.literal_eval(s) for s in env.mem_states])
            train_actions.extend(
                [{'same': 0, 'modify-1': 1, 'modify-2': 2, 'modify-3': 3}[a] for a in env.mem_action])
            train_rewards.extend(env.mem_reward)
            train_terminals.extend([0] * (len(env.mem_states) - 1) + [1])  # Last state as terminal

    # Convert lists to numpy arrays for compatibility with d3rlpy
    train_states = np.array(train_states)
    train_actions = np.array(train_actions)
    train_rewards = np.array(train_rewards)
    train_terminals = np.array(train_terminals)

    # Create MDP dataset for d3rlpy training
    traindataset = d3rlpy.datasets.MDPDataset(
        observations=train_states,
        actions=train_actions,
        rewards=train_rewards,
        terminals=train_terminals
    )

    # Prepare test user data (process test user separately)
    env.reset(True)
    env.process_data(test_user, 0)
    test_states = np.array([ast.literal_eval(s) for s in env.mem_states])
    test_actions = np.array(
        [{'same': 0, 'modify-1': 1, 'modify-2': 2, 'modify-3': 3}[a] for a in env.mem_action])
    test_rewards = np.array(env.mem_reward)
    test_terminals = np.array([0] * (len(test_states) - 1) + [1])  # Last state is terminal

    # Create MDP dataset for d3rlpy evaluation (test user data)
    testdataset = d3rlpy.datasets.MDPDataset(
        observations=test_states,
        actions=test_actions,
        rewards=test_rewards,
        terminals=test_terminals
    )

    # Prepare the Discrete BCQ algorithm for training
    cql = d3rlpy.algos.DiscreteCQLConfig().create()

    # Start training on the training users' dataset
    cql.fit(traindataset, n_steps=10000, n_steps_per_epoch=100, experiment_name=f"{d}_{task}")

    # Evaluate on the test user
    predictions = cql.predict(test_states)
    print(f"Predictions for {test_user}: {predictions}")

    # Calculate accuracy by comparing `predictions` to `test_actions`
    accuracy = np.mean(predictions == test_actions)
    print(f"Accuracy for {test_user}: {accuracy * 100:.2f}%")

    return accuracy, predictions, test_actions, test_user

def main():
    # Initialize environment
    env = environment5.environment_vizrec()
    datasets = ['movies']
    tasks = ['p1', 'p2', 'p3', 'p4']

    # Store overall accuracy across all users and tasks
    overall_accuracy = []

    for d in datasets:
        for task in tasks:
            print(f"# Dataset: {d}, Task: {task}")

            # Get the list of users for the dataset and task
            user_list_name = env.get_user_list(d, task)

            # Store accuracy for the current task
            task_accuracy = []
            detailed_results = []

            # Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_user, test_user, user_list_name, env, d, task) for test_user in user_list_name]
                for future in concurrent.futures.as_completed(futures):
                    accuracy, predictions, true_actions, test_user = future.result()
                    task_accuracy.append(accuracy)
                    detailed_results.extend([(test_user, pred, true) for pred, true in zip(predictions, true_actions)])

            # Calculate and print the average accuracy for the current task
            average_task_accuracy = np.mean(task_accuracy)
            print(f"Average Accuracy for Task {task}: {average_task_accuracy * 100:.2f}%")

            # Append task accuracy to overall accuracy
            overall_accuracy.extend(task_accuracy)

            # Write detailed results to a CSV file
            with open(f"results_{d}_{task}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Test User', 'Prediction', 'True Action'])
                writer.writerows(detailed_results)

    # Calculate and print the overall accuracy across all tasks and users
    overall_accuracy_mean = np.mean(overall_accuracy)
    print(f"Overall Accuracy: {overall_accuracy_mean * 100:.2f}%")

if __name__ == "__main__":
    main()
