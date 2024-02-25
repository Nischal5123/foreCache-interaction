import numpy as np
import os
import pandas as pd
from collections import Counter, defaultdict
import csv
from itertools import product
import ast
import json

class InteractionProcessor:
    def __init__(self, user_interactions_path, processed_interactions_path,master_data_path,imp_attrs):
        self.user_interactions_path = user_interactions_path
        self.processed_interactions_path = processed_interactions_path
        self.master_data_path = master_data_path
        self.fieldnames = ['Airport_Name', 'Aircraft_Make_Model', 'Effect_Amount_of_damage', 'Flight_Date', 'Aircraft_Airline_Operator', 'Origin_State', 'When_Phase_of_flight', 'Wildlife_Size', 'Wildlife_Species', 'When_Time_of_day', 'Cost_Other', 'Cost_Repair', 'Cost_Total', 'Speed_IAS_in_knots','None']
        self.max_len=len(self.fieldnames)
        self.bookmarks=0
        self.important_attributes_counter=imp_attrs
        self.important_attributes_counter_per_user=imp_attrs

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



    def get_final_bookmark_vglstr(self,csv_filename):
        path=self.master_data_path+'logs/'
        json_file_name = csv_filename.split('_logs.csv')[0] + '_bookmarked.json'
        with open(os.path.join(path, json_file_name)) as file:
            data = json.load(file)
        # Extract all keys
        all_vglstr = list(data.keys())
        # Extract the last key
        return all_vglstr

    def user_specific_reward(self, csv_filename):
        # Initialize counter dictionary if not already initialized
        if not self.important_attributes_counter_per_user:
            self.important_attributes_counter_per_user = {field: 0 for field in self.fieldnames}

        # Get bookmark reward
        bookmarked_vglstr_strings = self.get_final_bookmark_vglstr(csv_filename)
        print('User:', csv_filename, 'Total charts bookmarked:', len(bookmarked_vglstr_strings))

        for bookmark_string in bookmarked_vglstr_strings:
            for field in self.fieldnames:
                if field in bookmark_string:
                    print(f'Field "{field}" found in bookmark string: {bookmark_string}')
                    self.important_attributes_counter_per_user[field] += 1

        return self.important_attributes_counter_per_user


    def process_interaction_logs(self, csv_filename):
        print("Converting '{}'...".format(csv_filename))
        original_df = pd.read_csv(os.path.join(self.user_interactions_path, csv_filename))
        print('Total size of interaction log', len(original_df))
        df = pd.DataFrame(columns=[['User_Index', 'Interaction', 'Value', 'Time', 'Reward', 'Action', 'Attribute', 'State',
                 'High-Level-State']])
        df_attributes = []
        df_user_index = []
        df_reward = []
        df_action = []
        df_state = []
        df_high_state = []
        df_time = []
        df_interaction =[]
        df_value = []
        prev_time_hover = 0
        prev_time_scroll = 0
        for index, row in original_df.iterrows():

            value = row['Value']
            interaction = row['Interaction']
            attributes = []
            if pd.isna(value): #handle nulls
                pass
            else:
                if 'added chart to bookmark' in interaction or 'main chart changed' in interaction or 'clicked on a field' in interaction:
                    try:
                        attributes = self.get_fields_from_vglstr_updated(value)
                    except IndexError as a:
                        attributes=[]
                        print(a)
                        for attrs in self.fieldnames:
                            if attrs in value:
                                attributes.append(attrs)
                                #make sure final length is 3 else append with None
                        while len(attributes) < 3:
                            attributes.append('None')

                if 'mouse' in interaction:
                    time_period = (row['Time'] - prev_time_hover) / 1000
                    if time_period > 0.5 and value:
                        attributes = self.get_fields_from_vglstr_updated(value)
                    prev_time_hover = row['Time']



                if 'scroll' in interaction:
                    time_period = (row['Time'] - prev_time_scroll) / 1000
                    if time_period > 0.5 and value:
                        attributes = self.get_fields_from_vglstr_updated(value)
                    prev_time_scroll = row['Time']



                if len(attributes) != 0:
                    df_attributes.append(sorted(attributes))

                    # Calculate reward based on common attributes between 'attributes' and 'important_attributes'

                    reward = 0.1
                    for attribute in attributes:
                        try:
                            reward += self.important_attributes_counter_per_user[attribute]
                        except KeyError as e:
                            print(e)
                            reward +=0

                    action, _, _ = self.get_action_reward(interaction)
                    df_action.append(action)
                    df_reward.append(reward)
                    df_state.append(self.one_hot_encode_state(df_attributes[-1]))
                    df_high_state.append(self.get_state(interaction))
                    df_user_index.append(index)
                    df_time.append(row['Time'])
                    df_interaction.append(interaction)
                    df_value.append(value)

        df['Attribute'] = df_attributes
        df['Reward'] = df_reward
        df['Action'] = df_action
        df['State'] = df_state
        df['High-Level-State'] = df_high_state
        df['Time']=df_time
        df['Interaction']=df_interaction
        df['Value']=df_value
        df['User_Index'] = df_user_index  # Use the existing DataFrame index as User_Index

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

            num_different_elements = sum(c1 != c2 for c1, c2 in zip(sorted(current_state), sorted(next_state)))

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


    def get_user_name(self,url):
        parts = url.split('/')
        fname = parts[-1]
        uname = fname.rstrip('_log.csv')
        return uname

    def create_master_file(self, csvfiles,task,isbirdstrike=False):
        if isbirdstrike:
            master_csv_filename = task + '-combined-interactions-birdstrike.csv'
        else:
            master_csv_filename = task +'-combined-interactions.csv'
        dfs = []  # List to store individual DataFrames

        for csv_filename in csvfiles:
            end = task + '_logs.csv'
            if csv_filename.endswith(end):
                # print("Converting '{}'...".format(csv_filename))
                df = pd.read_csv(os.path.join(self.processed_interactions_path, csv_filename))

                # Add a new column 'User' and populate it with the original name of the small file
                df['User'] = self.get_user_name(csv_filename)

                # Drop the column named 'User_Index'
                df.drop('User_Index', axis=1, inplace=True)

                # Reset the index for each individual file and create a new index column 'U_Index'
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'U_Index'}, inplace=True)
                # Calculate the mean time for this DataFrame
                mean_time = df['Time'].mean()

                # Create the 'Trial' column based on the condition for this DataFrame
                df['Trial'] = df['Time'].apply(lambda x: 1 if x < mean_time else 2)

                dfs.append(df)

        # Concatenate individual DataFrames into a single master DataFrame
        master_df = pd.concat(dfs, ignore_index=True)

        # Save the master DataFrame to a CSV file
        master_csv_path = os.path.join(self.master_data_path, master_csv_filename)
        master_df.to_csv(master_csv_path, index=False)





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




def get_base_reward(csv_files,path,t='p2'):
    # Initialize a default dictionary with 0 values for each field name
    important_attributes_counter_across_user = defaultdict(lambda: 0)
    task= t+ '_logs.csv'
    fields = ['Airport_Name', 'Aircraft_Make_Model', 'Effect_Amount_of_damage', 'Flight_Date',
              'Aircraft_Airline_Operator', 'Origin_State', 'When_Phase_of_flight', 'Wildlife_Size',
              'Wildlife_Species', 'When_Time_of_day', 'Cost_Other', 'Cost_Repair', 'Cost_Total',
              'Speed_IAS_in_knots']
    users=0
    for csv_filename in csv_files:
        important_attributes_counter_user = defaultdict(lambda: 0)
        if csv_filename.endswith(task):
            users+=1
            print("Converting '{}'...".format(csv_filename))
            user_name = csv_filename.split('.')[0]

            df = pd.read_csv(os.path.join(path, csv_filename))
            print('Total size of interaction log', len(df))

            #number of users using attribute in the task
            for index, row in df.iterrows():
                for field in fields:
                    if field in row['Value']:
                        important_attributes_counter_user[field] += 1

        #get across useer count for thoise fileds which are > 1 per user
        for field in fields:
            if important_attributes_counter_user[field] > 0:
                important_attributes_counter_across_user[field] += 1


    # divide by number of users
    normalized_important_attributes = {key: value / users for key, value in important_attributes_counter_across_user.items()}
    return normalized_important_attributes


def remove_invalid_rows(user_path, csv_filename):

        df = pd.read_csv(os.path.join(user_path, csv_filename))
        # Drop rows where 'Interaction' contains 'typed in answer'
        df = df[~df['Interaction'].str.contains('typed in answer')]
        # Drop rows where 'Interaction' contains 'changed ptask ans'
        df = df[~df['Interaction'].str.contains('changed ptask ans')]
        df = df[~df['Interaction'].str.contains('window')]
        df = df[~df['Interaction'].str.contains('study begins')]
        #remove all rows with null in 'Value' column
        df = df.dropna(subset=['Value'])
        df = df.reset_index(drop=True)

        prev_time_hover = 0
        prev_time_scroll = 0

        for index, row in df.iterrows():
            interaction=row['Interaction'] = row['Interaction']

            if 'bookmark' in interaction or 'main chart changed' in interaction or 'clicked on a field' in interaction:
                pass

            elif 'mouse' in interaction:
                time_period = (row['Time'] - prev_time_hover) / 1000
                if time_period < 0.5:
                    #drop the row
                    df.drop(index, inplace=True)
                    prev_time_hover = row['Time']

            elif 'scroll' in interaction:
                time_period = (row['Time'] - prev_time_scroll) / 1000
                if time_period < 0.5 :
                    #drop the row
                    df.drop(index, inplace=True)
                    prev_time_scroll = row['Time']

            else:
                pass

        df.to_csv(os.path.join(user_path, csv_filename), index=False)

if __name__ == '__main__':
    task = 'p1'
    user_interactions_path = './data/zheng/birdstrikes_processed_csv/'
    csv_files = os.listdir(user_interactions_path)
    for csv_filename in csv_files:
        end=task+'_logs.csv'
        if csv_filename.endswith(end):
            remove_invalid_rows(user_interactions_path, csv_filename)

    important_attrs = get_base_reward(csv_files,user_interactions_path,task)
    print('Important attributes:', important_attrs, 'for task:', task, 'length:', len(important_attrs))

    processed_interactions_path = './data/zheng/birdstrikes_processed_interactions_'+task
    master_data_path ='./data/zheng/'



    for csv_filename in csv_files:
        end = task + '_logs.csv'
        if csv_filename.endswith(end):
             interaction_processor = InteractionProcessor(user_interactions_path, processed_interactions_path, master_data_path,important_attrs)
             interaction_processor.user_specific_reward(csv_filename)
             interaction_processor.process_interaction_logs(csv_filename)
             interaction_processor.process_actions(csv_filename)



