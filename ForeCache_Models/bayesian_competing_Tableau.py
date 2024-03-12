import pandas as pd
import ast
import numpy as np
import itertools
from Reward_Generator import reward
from read_data import read_data
from sklearn.preprocessing import LabelEncoder
import pdb
import random
from Categorizing_v4 import Categorizing
from itertools import combinations
import random


class ExtendedCategoricalModel:
    def __init__(self, data, action_var_name, context_var_names, alpha):
        self.data = data
        self.action_var_name = action_var_name
        self.context_var_names = context_var_names
        self.alpha = alpha

        self.actions = [0, 1, 2, 3, 4]
        self.context_categories = {var_name: list(data[var_name].unique()) for var_name in context_var_names}
        # pdb.set_trace()
        self.counts = self.init_counts()
        self.probabilities = self.update_probabilities()

    def init_counts(self):
        # Initialize counts for each combination of context and action to alpha for smoothing
        counts = {}
        for context_combination in itertools.product(
                *[self.context_categories[var_name] for var_name in self.context_var_names]):
            counts[context_combination] = {}
            for action in self.actions:
                counts[context_combination][action] = self.alpha
        return counts

    def update_probabilities(self):
        # Update probabilities based on counts
        probabilities = {}
        for context_combination, actions in self.counts.items():
            probabilities[context_combination] = {}
            for action, count in actions.items():
                total_count = sum(self.counts[context_combination][a] for a in self.actions)
                probabilities[context_combination][action] = count / total_count

        return probabilities

    def update(self, observation):
        # Update counts and probabilities based on a new observation
        action = observation[self.action_var_name]
        context_combination = tuple(observation[var_name] for var_name in self.context_var_names)
        self.counts[context_combination][action] += 1
        self.probabilities = self.update_probabilities()

    def predict_next_action(self, current_context):
        context_combination = tuple(current_context[var_name] for var_name in self.context_var_names)
        # Check if the context combination exists in our probabilities
        if context_combination in self.probabilities:
            action_probabilities = self.probabilities[context_combination]
        else:
            # If the context_combination is not present, initialize probabilities to uniform distribution
            return random.choice([0, 1, 2, 3, 4])
            # pdb.set_trace()
        if random.random() < action_probabilities[max(action_probabilities, key=action_probabilities.get)]:
            return max(action_probabilities, key=action_probabilities.get)
        else:
            return random.choice([0, 1, 2, 3, 4])

        # [0.4, 0.5, 0, 0.1]

    # return max(action_probabilities, key=action_probabilities.get)


def encoding(dataset, file):
    all_attributes = []
    all_actions = set()

    data_dataframe = []
    cat = Categorizing(dataset)

    if dataset == 'birdstrikes':
        states = {"Damage": 0, "Incident": 1, "Aircraft": 2, "Environment": 3, "Wildlife": 4, "Misc": 5}
        # self.states = {"Damage":0, "Incident":1, "Aircraft":2, "Environment":3}
    elif dataset == 'weather1':
        states = {"Temperature": 0, "Location": 1, "Metadata": 2, "CommonPhenomena": 3, "Fog": 4, "Extreme": 5,
                  "Misc": 6, "Misc2": 7}
    else:  # 'FAA1'
        states = {"Performance": 0, "Airline": 1, "Location": 2, "Status": 3, "Misc": 4}

    unique_attributes = set(states.keys())
    # attribute_encoder = LabelEncoder().fit(list(unique_attributes))
    all_combinations = []
    for r in range(1, 6):  # From 1 attribute up to 5
        for combo in combinations(unique_attributes, r):
            sorted_combo = tuple(sorted(combo))  # Sort the combination to ensure consistent ordering
            all_combinations.append(sorted_combo)
    print(len(all_combinations))
    # Encode all possible combinations
    attribute_encoder = LabelEncoder().fit([''.join(combo) for combo in all_combinations])

    for line in file:
        parts = line.split(",")
        parts[1] = parts[1].replace(";", ",")
        list_part = ast.literal_eval(parts[1])
        if len(list_part) == 0:
            continue
        high_level_attrs = cat.get_category(list_part, dataset)
        # print(high_level_attrs)
        x = attribute_encoder.transform(high_level_attrs)
        encoded_interaction = ''.join(str(p) for p in sorted(x))
        data_dataframe.append([int(parts[0]), encoded_interaction])

    data = pd.DataFrame(data_dataframe, columns=['Action', 'Attribute'])

    data['Encoded_Attribute'] = attribute_encoder.fit_transform(data['Attribute'])
    action_encoder = LabelEncoder()
    data['Action'] = action_encoder.fit_transform(data['Action'])
    # pdb.set_trace()
    # print(data['Action'])
    return data


obj = read_data()
obj.create_connection(r"Tableau.db")
r = reward()
final_results = np.zeros(9, dtype=float)
for d in r.datasets:
    users = obj.get_user_list_for_dataset(d)
    # getting all users data for model initialization
    cleaned_data = []
    for user in users:
        u = user[0]
        data = obj.merge2(d, u)
        raw_states, raw_actions, mem_reward = r.generate(data, d)
        for i in range(len(raw_states)):
            temp = str(raw_actions[i]) + ",["
            for idx, s in enumerate(raw_states[i]):
                if idx == len(raw_states[i]) - 1:
                    temp += s
                else:
                    temp += s + ";"
            temp += "]"
            cleaned_data.append(temp)
            # print(temp)
            # pdb.set_trace()
    all_user = encoding(d, cleaned_data)

    for user in users:
        u = user[0]
        data = obj.merge2(d, u)
        raw_states, raw_actions, mem_reward = r.generate(data, d)
        cleaned_data = []
        for i in range(len(raw_states)):
            temp = str(raw_actions[i]) + ",["
            # print(raw_actions[i], end = ",[")
            for idx, s in enumerate(raw_states[i]):
                if idx == len(raw_states[i]) - 1:
                    # print(s, end="")
                    temp += s
                else:
                    # print(s, end=";")
                    temp += s + ";"
            # print("]",)
            temp += "]"
            cleaned_data.append(temp)

        data2 = encoding(d, cleaned_data)

        #data2 :# action , attribute (as string from array) , number of that combination

        #all_user :

        model = ExtendedCategoricalModel(all_user, 'Action', ['Encoded_Attribute'], alpha=1)

        # model = ExtendedCategoricalModel(data2, 'Action', ['Encoded_Attribute'], alpha=1)

        # # Update the model with observations (you'd loop through your observations to update)
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            accu = []
            split = int(len(data2) * t)
            for idx in range(split):
                # print(data2.iloc[idx])
                model.update(data2.iloc[idx])
            for idx in range(split, len(data2)):
                predicted_action = model.predict_next_action(data2.iloc[idx])
                if predicted_action == data2['Action'][idx]:
                    accu.append(1)
                else:
                    accu.append(0)
                model.update(data2.iloc[idx])
            results.append(np.mean(accu))
            # print(t, np.mean(results))
        final_results = np.add(final_results, results)
        # pdb.set_trace()
        # print(len(raw_states), len(raw_actions), len(mem_reward))
    final_results /= len(users)
    print(d, ", ".join(f"{x:.2f}" for x in final_results))

# # Initialize an empty list to store the parsed data
# data2 = []

# # Open the file and read each line
# with open("example_data.txt", 'r') as file:
#     for line in file:
#         z = ast.literal_eval(line)
#         # print(z)
#         data2.append(z)

# data = pd.DataFrame(data2, columns=['id', 'time', 'action', 'visualization', 'state', 'reward'])

# # print(df.head())  #

# model = ExtendedCategoricalModel(data, 'action', ['visualization', 'state'], alpha=1)
# # Update the model with observations (you'd loop through your observations to update)
# threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# for t in threshold:
#     results = []
#     split = int(len(data) * t)
#     for idx in range(split):
#         model.update(data.iloc[idx])
#     for idx in range(split, len(data)):
#         predicted_action = model.predict_next_action(data.iloc[idx])
#         if predicted_action == data['action'][idx]:
#             results.append(1)
#         else:
#             results.append(0)
#     print(t, np.mean(results))