import environment_vizrec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ANALYSIS_FOLDER = 'Experiments_Folder/VizRec/Analysis/'

import sklearn.tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def decision_tree_analysis(birdstrikesActions, moviesActions, analysis_folder):
    # Create n-gram model to extract common action combinations (n-grams)
    n = 2  # bi-grams
    birdstrikes_ngrams = [tuple(birdstrikesActions[i:i + n]) for i in range(len(birdstrikesActions) - n + 1)]
    movies_ngrams = [tuple(moviesActions[i:i + n]) for i in range(len(moviesActions) - n + 1)]

    # Combine actions from both datasets and add labels to identify them
    combined_ngrams = birdstrikes_ngrams + movies_ngrams
    dataset_labels = ['birdstrikes'] * len(birdstrikes_ngrams) + ['movies'] * len(movies_ngrams)

    # Encode features and labels as numerical values
    encoder = LabelEncoder()

    # Flatten the tuples of n-grams for encoding
    flat_features = [str(ngram) for ngram in combined_ngrams]
    encoded_features = encoder.fit_transform(flat_features).reshape(-1, 1)  # Reshape for classifier

    # Encode dataset labels (birdstrikes vs. movies)
    encoded_labels = encoder.fit_transform(dataset_labels)

    # Train the decision tree classifier
    model = sklearn.tree.DecisionTreeClassifier()
    model.fit(encoded_features, encoded_labels)

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(20, 10))
    # sklearn.tree.plot_tree(model, ax=ax, feature_names=['Action Patterns'], class_names=encoder.classes_, filled=True)
    #feature_names should be original n-gram names
    sklearn.tree.plot_tree(model, ax=ax, feature_names=flat_features, class_names=encoder.classes_, filled=True)
    plt.savefig(f'{analysis_folder}/decision_tree_comparison.png')
    plt.show()

    print("Decision tree comparing action patterns saved as decision_tree_comparison.png")


# Example usage:
# decision_tree_analysis(birdstrikesActions, moviesActions, '/path/to/analysis_folder


def process_all_users(dataset, all_dataset_users):
    all_actions = []
    print(all_dataset_users)
    for user in all_dataset_users:
        env = environment_vizrec.environment_vizrec()
        env.process_data(user,0)
        for i in range(1, len(env.mem_action)):
            all_actions.append(env.mem_action[i])
    unique, counts = np.unique(all_actions, return_counts=True)
    #get percentage of each action
    action_counts = dict(zip(unique, counts))
    action_percentages = {k: v / len(all_actions) for k, v in action_counts.items()}
    print(f'Action distribution for {dataset} is {action_percentages}')
    #get transition matrix for each action to another action
    action_transitions = {}
    for action in unique:
        action_transitions[action] = {}
        for action2 in unique:
            action_transitions[action][action2] = 0
    #conscutive actions +1
    for i in range(1, len(all_actions)-1):
        action_transitions[all_actions[i]][all_actions[i+1]] += 1
    #get percentage of each transition
    for action in unique:
        total_transitions = sum(action_transitions[action].values())
        if total_transitions > 0:
            action_transitions[action] = {k: v / total_transitions for k, v in action_transitions[action].items()}


    #save both as csv to analysis folder
    action_percentages_df = pd.DataFrame.from_dict(action_percentages, orient='index', columns=['Percentage'])
    action_percentages_df.to_csv(f'{ANALYSIS_FOLDER}{dataset}_action_percentages.csv')

    #draw a heatmap for action transitions
    action_transitions_df = pd.DataFrame.from_dict(action_transitions, orient='index', columns=unique)
    action_transitions_df.to_csv(f'{ANALYSIS_FOLDER}{dataset}_action_transitions.csv')
    print(f'Action transitions for {dataset}')
    print(action_transitions_df)

    return all_actions


def process_all_user_states(dataset, all_dataset_users):
    all_states = []
    for user in all_dataset_users:
        env = environment_vizrec.environment_vizrec()
        env.process_data(user, 0)
        for i in range(1, len(env.mem_states)):
            all_states.append(env.mem_states[i])
    unique, counts = np.unique(all_states, return_counts=True)
    state_counts = dict(zip(unique, counts))
    state_percentages = {k: v / len(all_states) for k, v in state_counts.items()}
    #sort states by percentage
    state_percentages = dict(sorted(state_percentages.items(), key=lambda item: item[1], reverse=True))
    print(f'State distribution for {dataset} is {state_percentages}')






if __name__ == "__main__":
    datasets = ['movies', 'birdstrikes']
    tasks = ['p1', 'p2', 'p3', 'p4']

    overall_accuracy= []
    birdstrikesActions = []
    for dataset in datasets:
        dataset_acc = []
        all_dataset_users = []
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = sorted(env.get_user_list(dataset, task))
            for user in user_list_name:
                all_dataset_users.append(user)
        all_actions=process_all_users(dataset, all_dataset_users)
        process_all_user_states(dataset, all_dataset_users)
        if dataset == 'birdstrikes':
            birdstrikesActions = all_actions
        else:
            moviesActions = all_actions
    decision_tree_analysis(birdstrikesActions, moviesActions, ANALYSIS_FOLDER)
