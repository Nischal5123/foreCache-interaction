from monadjemi_competing_models import CompetingModels
import pandas as pd
import csv
import itertools
import os
import ast
import numpy
import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")
def create_underlying_data():
    x_attributes = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date', 'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type', 'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes', 'None']

    # Use the same list for all three sets of attributes
    y_attributes = x_attributes
    z_attributes = x_attributes

    # Open a CSV file for writing
    with open('./data/zheng/combinations.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(['id', 'x_attribute', 'y_attribute', 'z_attribute'])

        # Initialize an ID counter
        id_counter = 0

        # Write all combinations
        for combination in itertools.product(x_attributes, repeat=3):
            writer.writerow([id_counter] + list(combination))
            id_counter += 1

    print("CSV file 'combinations.csv' has been created.")

def get_interaction_id(search_entry):
    search_entry_list = ast.literal_eval(search_entry)

    # Open the CSV file and search for the entry
    with open('./data/zheng/combinations.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)

        # Skip the header row
        next(reader)

        # Iterate through the rows
        for row in reader:
            row_values = row[1:]  # Exclude the 'id' column
            if sorted(row_values) == sorted(search_entry_list):
                found_id = row[0]
                return found_id

    # Return a default value if the entry is not found
    return 4912

if __name__ == '__main__':
    create_underlying_data()
    processed_interactions_path = 'data/zheng/processed_interactions_p4/'
    master_data_path = './data/zheng/'
    csv_files = os.listdir(processed_interactions_path)
    output_file_path='./data/zheng/results.csv'

    combined_user_interactions = pd.DataFrame(columns=['user', 'interaction_session'])
    for csv_filename in csv_files:
        print("Converting '{}'...".format(csv_filename))
        user_name = csv_filename.split('.')[0]
        df = pd.read_csv(os.path.join(processed_interactions_path, csv_filename))
        interactions = []
        print('Total size of interaction log', len(df))
        for index, row in df.iterrows():
            value = row['Attribute']
            match_to_dataset = int(get_interaction_id(value))
            interactions.append(match_to_dataset)
        print(user_name, len(interactions))
        combined_user_interactions = pd.concat([combined_user_interactions, pd.DataFrame({'user': [user_name], 'interaction_session': [interactions]})], ignore_index=True)
        interactions = []
    print("Conversion complete.")
    combined_user_interactions.to_csv("./data/zheng/competing_movies_interactions.csv", index=False)


    underlying_data = pd.read_csv("./data/zheng/combinations.csv")
    underlying_data.set_index("id", drop=True, inplace=True)
    d_attrs = ["x_attribute", "y_attribute","z_attribute"]

    interaction_data = pd.read_csv('./data/zheng/competing_movies_interactions.csv')
    # Filter rows where 'user' column ends with 'p4_logs'
    interaction_data = interaction_data[interaction_data['user'].str.endswith('p4_logs')]
    interaction_data['interaction_session'] = interaction_data.apply(
        lambda row: ast.literal_eval(row.interaction_session), axis=1)

    for participant_index, row in interaction_data.iterrows():
        print(f'Processing user {row.user}')
        results = {'participant_id': row.user}
        competing_models = CompetingModels(underlying_data, [], d_attrs)
        predicted = pd.DataFrame()
        rank_predicted = []

        for i in range(len(interaction_data.iloc[participant_index].interaction_session)):
            interaction = interaction_data.iloc[participant_index].interaction_session[i]
            competing_models.update(interaction)
            thres=0.50
            if i > (thres*len(interaction_data.iloc[participant_index].interaction_session)) - 1 and i < len(interaction_data.iloc[participant_index].interaction_session) - 1:
                probability_of_next_point = competing_models.predict()
                next_point = interaction_data.iloc[participant_index].interaction_session[i + 1]
                predicted_next_dict = {}
                predicted_next_dict[1] = (next_point in probability_of_next_point.nlargest(1).index.values)
                predicted = pd.concat([predicted, pd.DataFrame([predicted_next_dict])], ignore_index=True)
                sorted_prob = probability_of_next_point.sort_values(ascending=False)
                rank, = np.where(sorted_prob.index.values == next_point)
                rank_predicted.append(rank[0] + 1)

        ncp = predicted.sum() / len(predicted)
        results['rank'] = rank_predicted

        for col in ncp.index:
            results[f'ncp-{col}'] = ncp[col]

        bias = competing_models.get_attribute_bias()
        for col in bias.columns:
            results[f'bias-{col}'] = bias[col].to_numpy()

        posterior = competing_models.get_model_posterior()
        for col in posterior.columns:
            results[f'posterior-{col}'] = posterior[col].to_numpy()

        # Append the results to the zeng_map_results DataFrame
        zeng_map_results = zeng_map_results.append(results, ignore_index=True)
        zeng_map_results.csv(output_file_path)