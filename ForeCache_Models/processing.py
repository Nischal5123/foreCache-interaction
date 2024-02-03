import numpy as np
import os
import pandas as pd
from collections import Counter, defaultdict
import csv
from itertools import product
import ast

class InteractionProcessor:
    def __init__(self, user_interactions_path, processed_interactions_path,master_data_path,imp_attrs):
        self.user_interactions_path = user_interactions_path
        self.processed_interactions_path = processed_interactions_path
        self.master_data_path = master_data_path
        self.fieldnames = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date',
                           'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type',
                           'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes','None']
        self.max_len=len(self.fieldnames)
        self.bookmarks=0
        self.important_attributes_counter=imp_attrs

    def get_fields_from_vglstr(self, vglstr):
        encoding_str = vglstr.split(';')[1]
        encoding_str = encoding_str.split(':')[1]
        encodings = encoding_str.split(',')
        fields = []
        for encode in encodings:
            field = encode.split('-')[0]
            if field == '':
                pass
            fields.append(field)
        return sorted(fields)

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
        return sorted(fields)

    # def get_fields_from_vglstr_updated(self, vglstr):
    #     encoding_str = vglstr.split(';')[1]
    #     encoding_str = encoding_str.split(':')[1]
    #     encodings = encoding_str.split(',')
    #     fields = []
    #     for encode in encodings:
    #         field = encode.split('-')[0]
    #         if field == '':
    #             continue
    #         else:
    #                 fields.append(field)
    #     return sorted(fields)

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

    # def one_hot_encode_state(self, attributes):
    #     # Initialize a list of zeros with the length of fieldnames
    #     one_hot = [0] * self.max_len
    #
    #     # Set index corresponding to each attribute in the fieldnames
    #     for idx in range(len(attributes)):
    #         index = self.fieldnames.index(attributes[idx])
    #         one_hot[index] = 1
    #
    #     return one_hot

    def one_hot_encode_state(self, attributes):
        # Initialize a list of zeros with the length of fieldnames
        one_hot = [0] * 3

        # Set index corresponding to each attribute in the fieldnames
        for idx in range(len(attributes)):
            index = self.fieldnames.index(attributes[idx])
            one_hot[idx] = index

        return one_hot

    def create_combination_file(self):

        master_comb_filename='possible-combinations.csv'

        z_attributes,y_attributes,x_attributes =self.fieldnames,self.fieldnames, self.fieldnames

        # Open a CSV file for writing
        master_csv_path = os.path.join(self.master_data_path, master_comb_filename)
        with open(master_csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row
            writer.writerow(['id', 'z_attribute', 'x_attribute', 'y_attribute'])

            # Initialize an ID counter
            id_counter = 0

            # Write all combinations
            for combination in product(z_attributes, x_attributes, y_attributes):
                z_attr, x_attr, y_attr = sorted(combination)

                # Ensure that each attribute is unique
                if len(set([z_attr, x_attr, y_attr])) == 3:
                    writer.writerow([id_counter, z_attr, x_attr, y_attr])
                    id_counter += 1

        print("CSV file {} has been created.".format(master_comb_filename))

    def get_interaction_id(self,search_entry):

        # Open the CSV file and search for the entry
        master_comb_filename = 'possible-combinations.csv'
        master_csv_path = os.path.join(self.master_data_path, master_comb_filename)
        with open(master_csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)

            # Skip the header row
            next(reader)

            # Initialize a variable to store the found ID
            found_id = 4080

            # Iterate through the rows
            for row in reader:
                row_values = row[1:]  # Exclude the 'id' column
                if row_values == search_entry:
                    found_id = row[0]
                    break

        return found_id



    def process_interaction_logs(self, csv_filename):
        print("Converting '{}'...".format(csv_filename))
        df = pd.read_csv(os.path.join(self.user_interactions_path, csv_filename))
        print('Total size of interaction log', len(df))

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
            if type(value)==float: #handle nulls
                pass
            else:
                try:
                    attributes = self.get_fields_from_vglstr_updated(value)
                except IndexError as a:
                    print(a)
                    attributes = ['None', 'None', 'None']
            df_attributes.append(attributes)

            # Calculate reward based on common attributes between 'attributes' and 'important_attributes'
            reward = 0.1
            for attribute in attributes:
                try:
                    reward += self.important_attributes_counter[attribute]
                except KeyError as e:
                    print(e)
                    reward +=0
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

        for index in range(len(df) - 1):
            current_state = sorted(np.array(eval(df['State'][index])))  # Convert string representation to list
            next_state = sorted(np.array(eval(df['State'][index + 1])))  # Convert string representation to list
            action = ''

            num_different_elements = sum(c1 != c2 for c1, c2 in zip(current_state, next_state))

            if num_different_elements == 0:
                action = 'same'
            else:
                action = f'modify-{num_different_elements}'
                print('Reset')

            actions.append(action)

        actions.append('same')
        df['Action'] = actions

        # Save the modified DataFrame
        df.to_csv(os.path.join(self.processed_interactions_path, csv_filename), index=False)

    # def process_actions(self, csv_filename):
    #     df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))
    #     actions = []
    #
    #     consecutive_count = 1
    #
    #     for index in range(len(df) - 1):
    #         current_state = sorted(ast.literal_eval(df['State'][index]))
    #         next_state = sorted(ast.literal_eval(df['State'][index + 1]))
    #
    #         num_different_elements = sum(c1 != c2 for c1, c2 in zip(current_state, next_state))
    #
    #         if num_different_elements == 0:
    #             consecutive_count += 1
    #             if consecutive_count <= 3:
    #                 action = 'same'
    #             else:
    #                 action = None  # Do not append 'same' to the actions list
    #         else:
    #             action = f'modify-{num_different_elements}'
    #             consecutive_count = 1  # Reset consecutive count
    #
    #         actions.append(action)
    #
    #     actions.append('same')
    #     df['Action'] = actions
    #
    #     # Drop rows with 'None' in the 'Action' column
    #     df = df[df['Action'].notna()]
    #
    #     # Save the modified DataFrame
    #     df.to_csv(os.path.join(self.processed_interactions_path, csv_filename), index=False)
    def get_user_name(self,url):
        parts = url.split('/')
        fname = parts[-1]
        uname = fname.rstrip('_log.csv')
        return uname

    def create_master_file(self, csvfiles):
        master_csv_filename = 'combined-interactions.csv'
        dfs = []  # List to store individual DataFrames

        for csv_filename in csvfiles:
            if csv_filename.endswith('p4_logs.csv'):
                # print("Converting '{}'...".format(csv_filename))
                df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))

                # Add a new column 'User' and populate it with the original name of the small file
                df['User'] = self.get_user_name(csv_filename)

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

        # Sort DataFrame by 'Time' column, doesnt change oreder but anyway
        df.sort_values(by='Time', inplace=True)

        # Calculate the time difference between consecutive rows
        df['Time_Diff'] = df['Time'].diff()

        # Keep rows where the time difference is greater than or equal to 0.5 second
        df = df[df['Time_Diff'] >= 0.5]

        # Drop the 'Time_Diff' column as it is no longer needed
        df.drop(columns=['Time_Diff'], inplace=True)

        # Save the modified DataFrame
        df.to_csv(os.path.join(self.processed_interactions_path, csv_filename), index=False)



def get_fields_from_vglstr(vglstr):
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
def get_important_attributes(csv_files,path):
    # Initialize a default dictionary with 0 values for each field name
    important_attributes_counter = defaultdict(lambda: 0)

    important_attributes = []
    important_attributes_exact = []

    for csv_filename in csv_files:
        if csv_filename.endswith('p4_logs.csv'):
            print("Converting '{}'...".format(csv_filename))
            user_name = csv_filename.split('.')[0]

            df = pd.read_csv(os.path.join(path, csv_filename))
            print('Total size of interaction log', len(df))

            for index, row in df.iterrows():
                value = row['Value']
                interaction = row['Interaction']
                if 'bookmark' in interaction:
                    try:
                        attrs = get_fields_from_vglstr(value)
                        for a in attrs:
                            important_attributes_counter[a] += 1
                    except:
                        pass

    # Calculate the sum of all values
    total_count = sum(important_attributes_counter.values())
    # Normalize the Counter dictionary
    normalized_important_attributes = {key: value / total_count for key, value in important_attributes_counter.items()}

    print('#########  Number of charts added to bookmark #######', len(important_attributes_exact))
    return normalized_important_attributes

if __name__ == '__main__':
    user_interactions_path = './data/zheng/processed_csv/'
    processed_interactions_path = './data/zheng/processed_interactions_p4_bookmarked/'
    master_data_path ='./data/zheng/'
    csv_files = os.listdir(user_interactions_path)
    important_attrs=get_important_attributes(csv_files,user_interactions_path)
    interaction_processor = InteractionProcessor(user_interactions_path, processed_interactions_path,master_data_path,important_attrs)
    interaction_processor.create_combination_file()
    for csv_filename in csv_files:
        if csv_filename.endswith('p4_logs.csv'):
             interaction_processor.process_interaction_logs(csv_filename)
             interaction_processor.remove_invalid_rows(csv_filename)
             interaction_processor.process_actions(csv_filename)
             print(csv_filename)
    interaction_processor.create_master_file(csv_files)
    print(interaction_processor.important_attributes_counter)
