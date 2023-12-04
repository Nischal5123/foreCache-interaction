import numpy as np
import os
import pandas as pd
from collections import Counter

class InteractionProcessor:
    def __init__(self, user_interactions_path, processed_interactions_path,master_data_path):
        self.user_interactions_path = user_interactions_path
        self.processed_interactions_path = processed_interactions_path
        self.master_data_path = master_data_path
        self.fieldnames = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date',
                           'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type',
                           'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes', 'None']

    def get_fields_from_vglstr(self, vglstr):
        encoding_str = vglstr.split(';')[1]
        encoding_str = encoding_str.split(':')[1]
        encodings = encoding_str.split(',')
        fields = []
        for encode in encodings:
            field = encode.split('-')[0]
            if field == '':
                continue
            fields.append(field)
        return fields

    def get_fields_from_vglstr_updated(self, vglstr):
        encoding_str = vglstr.split(';')[1]
        encoding_str = encoding_str.split(':')[1]
        encodings = encoding_str.split(',')
        fields = ['None'] * 3
        for encode in encodings:
            field = encode.split('-')[0]
            if field == '':
                continue
            else:
                if "-x" in encode:
                    fields[0] = field
                elif "-y" in encode:
                    fields[1] = field
                else:
                    fields[2] = field
        return fields

    def get_action_reward(self, interaction):
        if interaction == "main chart changed because of clicking a field":
            action = 'click'
            reward = 1
            state = 'Data_View'
        elif interaction == "specified chart":
            action = 'specified'
            reward = 1
            state = 'Specified_View'
        elif interaction == "added chart to bookmark":
            action = 'bookmark'
            reward = 10
            state = 'Top_Panel'
        elif interaction.startswith('mouseover'):
            action = 'mouse'
            reward = 1
            state = 'Related_View'
        elif interaction.startswith('mouseout'):
            action = 'mouse'
            reward = 1
            state = 'Related_View'
        else:
            action = 'configuring'
            reward = 0.1
            state = 'Task_Panel'
        return action, reward, state

    def get_state(self, interaction):
        if "field" in interaction:
            state = 'Foraging'
        elif "specified chart" in interaction or 'bookmark' in interaction:
            state = 'Sensemaking'
        elif 'related chart' in interaction or 'mouseout' in interaction or 'mouseover' in interaction:
            state = 'Foraging'
        elif 'typed' in interaction or 'study begins' in interaction:
            state = 'Navigation'
        else:
            state = 'Navigation'
        return state

    def one_hot_encode_state(self, attributes):
        # Initialize a list of zeros with the length of fieldnames
        one_hot = [0] * len(attributes)

        # Set 1 at the index corresponding to each attribute in the fieldnames
        for idx in range(len(attributes)):
            index = self.fieldnames.index(attributes[idx])
            one_hot[idx] = index

        return one_hot

    def process_interaction_logs(self, csv_filename):
        print("Converting '{}'...".format(csv_filename))
        user_name = csv_filename.split('.')[0]
        # print(os.path.join(self.user_interactions_path, csv_filename))

        df = pd.read_csv(os.path.join(self.user_interactions_path, csv_filename))
        print('Total size of interaction log', len(df))

        important_attributes = []
        important_attributes_exact = []

        for index, row in df.iterrows():
            value = row['Value']
            interaction = row['Interaction']
            try:
                attrs = self.get_fields_from_vglstr(value)
                if 'added chart to bookmark' in interaction:
                    important_attributes_exact.append(str(attrs))
                    for a in attrs:
                        important_attributes.append(a)
            except:
                pass

        # print('Important attribute', important_attributes)
        important_attributes_counter = Counter(important_attributes)
        print('#########  Number of charts added to bookmark #######', len(important_attributes_exact))
        # print('Important attribute exact', important_attributes_exact)
        important_attributes_exact_counter = Counter(important_attributes_exact)

        df_attributes = []
        df_user_index = []
        df_reward = []
        df_action = []
        df_state = []
        df_high_state = []


        for index, row in df.iterrows():
            df_user_index.append(index)
            value = row['Value']
            interaction = row['Interaction']
            try:
                attributes = self.get_fields_from_vglstr_updated(value)
                unmodified_attributes = self.get_fields_from_vglstr(value)
                df_attributes.append(attributes)

                # Calculate reward based on common attributes between 'attributes' and 'important_attributes'
                reward = 0.1
                # if important_attributes_exact_counter[str(unmodified_attributes)] > 1:
                #     print('Exact Match')
                # reward += important_attributes_exact_counter[str(unmodified_attributes)] *3

                for attribute in attributes:
                    reward += important_attributes_counter[attribute]
            except:
                df_attributes.append(['None', 'None', 'None'])
                reward = 0.1
            action, _, _ = self.get_action_reward(interaction)
            df_action.append(action)
            df_reward.append(reward)
            df_state.append(self.one_hot_encode_state(df_attributes[-1]))
            df_high_state.append(self.get_state(interaction))

        df['Attribute'] = df_attributes
        df['Reward'] = df_reward
        df['Action'] = df_action
        df['State'] = df_state
        df['High-Level-State'] = df_high_state
        df['User_Index'] = df_user_index  # Use the existing DataFrame index as User_Index
        #df.set_index('User_Index', inplace=True)  # Set 'User_Index' as the index

        # Drop rows where 'Value' is empty
        df = df[df['Value'].notna()]
        # Drop rows where 'Interaction' contains 'typed in answer'
        df = df[~df['Interaction'].str.contains('typed in answer')]
        # Drop rows where 'Interaction' contains 'changed ptask ans'
        df = df[~df['Interaction'].str.contains('changed ptask ans')]
        df = df[~df['Interaction'].str.contains('main chart changed')]
        df = df[~df['Interaction'].str.contains('clicked on a field')]
        df = df[~df['Interaction'].str.contains('window')]
        df = df[~df['Interaction'].str.contains('study begins')]

        # Reorder columns
        df = df[['User_Index', 'Interaction', 'Value', 'Time', 'Reward', 'Action', 'Attribute', 'State',
                 'High-Level-State']]

        processed_csv_path = os.path.join(self.processed_interactions_path, csv_filename)
        df.to_csv(processed_csv_path, index=False)  # Use index=False to avoid the "Unnamed: 0" column

    def process_actions(self, csv_filename):
       # print("Converting '{}'...".format(csv_filename))

        df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))
        actions = []

        for index in range(len(df) - 1):  # Iterate up to the second-to-last row
            current_state = np.array(eval(df['State'][index]))  # Convert string representation to list
            next_state = np.array(eval(df['State'][index + 1]))  # Convert string representation to list
            action = ''

            if np.array_equal(next_state, current_state):
                action = 'same'
            else:
                if next_state[0] != current_state[0]:
                    action = 'modify-x'
                if next_state[1] != current_state[1]:
                    action = 'modify-y'
                if next_state[2] != current_state[2]:
                    action = 'modify-z'
                if next_state[1] != current_state[1] and next_state[2] != current_state[2]:
                    action = 'modify-y-z'
                if next_state[0] != current_state[0] and next_state[2] != current_state[2]:
                    action = 'modify-x-z'
                if next_state[0] != current_state[0] and next_state[1] != current_state[1]:
                    action = 'modify-x-y'
                if next_state[0] == current_state[0] and next_state[1] == current_state[1] and next_state[2] == current_state[2]:
                    action = 'same'
                if next_state[0] != current_state[0] and next_state[1] != current_state[1] and next_state[2] != current_state[2]:
                    action = 'modify-x-y-z'

            actions.append(action)

        actions.append('same')
        df['Action'] = actions

        # Save the modified DataFrame
        df.to_csv(os.path.join(self.processed_interactions_path, csv_filename), index=False)

    def create_master_file(self, csvfiles):
        master_csv_filename = 'combined-interactions.csv'
        dfs = []  # List to store individual DataFrames

        for csv_filename in csvfiles:
            if csv_filename.endswith('p4_logs.csv'):
                # print("Converting '{}'...".format(csv_filename))
                df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))

                # Add a new column 'User' and populate it with the original name of the small file
                df['User'] = os.path.splitext(os.path.basename(csv_filename))[0]

                # Drop the column named 'User_Index'
                df.drop('User_Index', axis=1, inplace=True)

                # Reset the index for each individual file and create a new index column 'U_Index'
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'U_Index'}, inplace=True)

                dfs.append(df)

        # Concatenate individual DataFrames into a single master DataFrame
        master_df = pd.concat(dfs, ignore_index=True)

        # Save the master DataFrame to a CSV file
        master_csv_path = os.path.join(self.master_data_path, master_csv_filename)
        master_df.to_csv(master_csv_path, index=False)

    def remove_invalid_rows(self, csv_filename):
        df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))

        # Sort DataFrame by 'Time' column
        df.sort_values(by='Time', inplace=True)

        # Calculate the time difference between consecutive rows
        df['Time_Diff'] = df['Time'].diff()

        # Keep rows where the time difference is greater than or equal to 0.5 second
        df = df[df['Time_Diff'] >= 500]

        # Drop the 'Time_Diff' column as it is no longer needed
        df.drop(columns=['Time_Diff'], inplace=True)

        # Save the modified DataFrame
        df.to_csv(os.path.join(self.processed_interactions_path, csv_filename), index=False)

if __name__ == '__main__':
    user_interactions_path = './data/zheng/processed_csv/'
    processed_interactions_path = './data/zheng/processed_interactions/'
    master_data_path ='./data/zheng/'
    csv_files = os.listdir(user_interactions_path)

    interaction_processor = InteractionProcessor(user_interactions_path, processed_interactions_path,master_data_path)

    for csv_filename in csv_files:
        if csv_filename.endswith('p4_logs.csv'):
             interaction_processor.process_interaction_logs(csv_filename)
             interaction_processor.remove_invalid_rows(csv_filename)
             interaction_processor.process_actions(csv_filename)
             print(csv_filename)
    interaction_processor.create_master_file(csv_files)
