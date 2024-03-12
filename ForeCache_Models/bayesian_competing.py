import pandas as pd
import ast
import numpy as np 
import itertools
import json
import os
import random
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

class ExtendedCategoricalModel:
    def __init__(self, data, action_var_name, context_var_names, alpha):
        self.data = data
        self.action_var_name = action_var_name
        self.context_var_names = context_var_names
        self.alpha = alpha

        self.actions = [0, 1, 2, 3]
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
            return random.choice([0, 1, 2, 3])
            # pdb.set_trace()
        if random.random() < action_probabilities[max(action_probabilities, key=action_probabilities.get)]:
            return max(action_probabilities, key=action_probabilities.get)
        else:
            return random.choice([0, 1, 2, 3])

        # [0.4, 0.5, 0, 0.1]

    # return max(action_probabilities, key=action_probabilities.get)


# def encoding(dataset, file):
#     all_attributes = []
#     all_actions = set()
#
#     data_dataframe = []
#
#     if dataset == 'movies':
#         columns = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date',
#                    'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type',
#                    'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes', 'None']
#
#         # Create a dictionary with numbered values
#         states = {index: value for index, value in enumerate(columns)}
#     else :
#         columns = ['Airport_Name', 'Aircraft_Make_Model', 'Effect_Amount_of_damage', 'Flight_Date', 'Aircraft_Airline_Operator', 'Origin_State', 'When_Phase_of_flight', 'Wildlife_Size', 'Wildlife_Species', 'When_Time_of_day', 'Cost_Other', 'Cost_Repair', 'Cost_Total', 'Speed_IAS_in_knots','None']
#         states = {index: value for index, value in enumerate(columns)}
#
#
#     unique_attributes = set(states.keys())
#     # attribute_encoder = LabelEncoder().fit(list(unique_attributes))
#     all_combinations = []
#     for r in range(1, 6):  # From 1 attribute up to 5
#         for combo in combinations(unique_attributes, r):
#             sorted_combo = tuple(sorted(combo))  # Sort the combination to ensure consistent ordering
#             all_combinations.append(sorted_combo)
#     print(len(all_combinations))
#     # Encode all possible combinations
#     attribute_encoder = LabelEncoder().fit([''.join(combo) for combo in all_combinations])
#
#     for line in file:
#         parts = line.split(",")
#         parts[1] = parts[1].replace(";", ",")
#         list_part = ast.literal_eval(parts[1])
#         if len(list_part) == 0:
#             continue
#         high_level_attrs = cat.get_category(list_part, dataset)
#         # print(high_level_attrs)
#         x = attribute_encoder.transform(high_level_attrs)
#         encoded_interaction = ''.join(str(p) for p in sorted(x))
#         data_dataframe.append([int(parts[0]), encoded_interaction])
#
#     data = pd.DataFrame(data_dataframe, columns=['Action', 'Attribute'])
#
#     data['Encoded_Attribute'] = attribute_encoder.fit_transform(data['Attribute'])
#     action_encoder = LabelEncoder()
#     data['Action'] = action_encoder.fit_transform(data['Action'])
#     # pdb.set_trace()
#     # print(data['Action'])
#     return data


def create_underlying_and_user_data(user_file,username):
    # Create underlying data
    user_interactions_path = './data/zheng/processed_interactions_p4/'
    df_id=[]
    df_state=[]
    user_data_df=pd.read_csv(user_interactions_path+user_file)

    for i in range(len(user_data_df)):
        df_id.append(i)
        df_state.append(user_data_df['State'][i])
    underlying_data = pd.DataFrame({'id': df_id, 'state': df_state})
    user_interaction_data = pd.DataFrame({'user': [username], 'interaction_session': [df_id]})
    print("Underlying data created")
    return underlying_data, user_interaction_data

def user_location():
    task = 'p4'
    dataset = 'movies'
    user_interactions_path = './data/zheng/processed_csv/'
    csv_files = os.listdir(user_interactions_path)
    current_csv_files = []
    for csv_filename in csv_files:
        end = task + '_logs.csv'
        if csv_filename.endswith(end):
            current_csv_files.append(csv_filename)

    return current_csv_files


if __name__ == '__main__':
    hyperparam_file='sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    session=1
    dataset = 'movies'
    underlying_data_paths = {
        'movies': './data/zheng/combinations.csv'   }

    user_interaction_data_paths = {
        'movies': './data/zheng/competing_movies_interactions.csv'    }

    continuous_attributes = {
        'movies': []
    }

    discrete_attributes = {
        'movies': ['attribute1']
    }

    output_file_path = './output/movies/movies_results_test_hmm.pkl'


    underlying_data = pd.read_csv(underlying_data_paths[dataset])
    original_data=pd.read_csv(underlying_data_paths[dataset])
    underlying_data.set_index('id', drop=True, inplace=True)



    interaction_data = pd.read_csv(user_interaction_data_paths[dataset])[:1]
    interaction_data['interaction_session'] = interaction_data.apply(
        lambda row: ast.literal_eval(row.interaction_session), axis=1)


    # Not necessary to run if we already have results file for HMM
    # Running HMM through all user interaction sessions and saving results in file
    hmm_results = pd.DataFrame()
    ks= [1]
    all_threshold = hyperparams['threshold']

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(
        columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])
    results={}
    for participant_index, row in interaction_data.iterrows():
        print(f'Processing user {row.user}')
        user_data= interaction_data.iloc[participant_index].interaction_session
        results ['participant_id']= row.user
        length = len(user_data)
        for thres in all_threshold:
            threshold = int(length * thres)
            print("threshold", threshold, "length", length - 1)
            #uderlying_data is all possible states and actions
            model = ExtendedCategoricalModel(data, 'action', ['visualization', 'state'], alpha=1)
            predicted = pd.DataFrame()
            rank_predicted = []

obj = read_data()
obj.create_connection(r"Tableau.db")
r = reward()
results = np.zeros(9, dtype = float)
for d in r.datasets:
    users = obj.get_user_list_for_dataset(d)
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
            # print(temp)
        # bayes = bayesian()
        model = ExtendedCategoricalModel(data, 'action', ['visualization', 'state'], alpha=1)

        # print(bayes.run_bayesian(cleaned_data))
        z = bayes.run_bayesian(cleaned_data)
        results = np.add(results, z)
        # pdb.set_trace()
        # print(len(raw_states), len(raw_actions), len(mem_reward))
    results /= len(users)
    print(d, ", ".join(f"{x:.2f}" for x in results))

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