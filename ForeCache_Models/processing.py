import numpy as np
import os
import pandas as pd
from collections import Counter
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

def get_fields_from_vglstr_updated(vglstr):
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
                fields[0]=field
            elif "-y" in encode:
                fields[1]=field
            else:
                fields[2]=field

    return fields

def get_action_reward(interaction):
    if interaction == "main chart changed because of clicking a field":
        action='click'
        reward=1
        state='Data_View'
    elif interaction == "specified chart":
        action='specified'
        reward=1
        state='Specified_View'
    elif interaction == "added chart to bookmark":
        action='bookmark'
        reward=10
        state='Top_Panel'
    elif interaction.startswith('mouseover'):
        action='mouse'
        reward=1
        state='Related_View'
    elif interaction.startswith('mouseout'):
        action='mouse'
        reward=1
        state='Related_View'
    else:
        action='configuring'
        reward=0.1
        state='Task_Panel'
    return action, reward, state


def get_state(interaction):
    if "field" in interaction:
        state='Foraging'
    elif "specified chart" in interaction:
        state='Sensemaking'
    elif 'bookmark' in interaction:
        state='Sensemaking'
    elif 'related chart' in interaction:
        state='Foraging'
    elif 'typed' in interaction:
        state='Navigation'
    else:
        state='Navigation'
    return state

def one_hot_encode_state(attributes):
    fieldnames = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date', 'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type', 'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes', 'None']

    # Initialize a list of zeros with the length of fieldnames
    one_hot = [0] * len(attributes)

    # Set 1 at the index corresponding to each attribute in the fieldnames
    for idx in range(len(attributes)):
        index = fieldnames.index(attributes[idx])
        one_hot[idx] = index

    return one_hot




if __name__ == '__main__':
    user_interactions_path = './data/zheng/processed_csv/'
    processed_interactions_path = './data/zheng/processed_interactions/'
    csv_files = os.listdir(user_interactions_path)

    fieldnames = ['Title', 'US_Gross', 'Worldwide_Gross', 'US_DVD_Sales', 'Production_Budget', 'Release_Date',
                  'MPAA_Rating', 'Running_Time_min', 'Distributor', 'Source', 'Major_Genre', 'Creative_Type',
                  'Director', 'Rotten_Tomatoes_Rating', 'IMDB_Rating', 'IMDB_Votes', 'None']

    for csv_filename in csv_files:
        if csv_filename.endswith('p4_logs.csv'):
            print("Converting '{}'...".format(csv_filename))
            user_name = csv_filename.split('.')[0]
            print(os.path.join(user_interactions_path, csv_filename))

            df = pd.read_csv(os.path.join(user_interactions_path, csv_filename))
            print('Total size of interaction log', len(df))

            important_attributes = []
            important_attributes_exact = []
            for index, row in df.iterrows():
                value = row['Value']
                interaction = row['Interaction']
                try:
                    attrs = get_fields_from_vglstr(value)
                    if 'added chart to bookmark' in interaction:
                        important_attributes_exact.append(str(attrs))
                        for a in attrs:
                            important_attributes.append(a)
                except:
                    pass
            print('Important attribute', important_attributes)
            important_attributes_counter = Counter(important_attributes)
            print('Important attribute exact', important_attributes_exact)
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
                    attributes = get_fields_from_vglstr_updated(value)
                    df_attributes.append(attributes)

                    # Calculate reward based on common attributes between 'attributes' and 'important_attributes'
                    reward = 0.1
                    if important_attributes_exact_counter[str(attributes)] > 1:
                        print('Exact Match')
                    reward += important_attributes_exact_counter[str(attributes)] * 3
                    for attribute in attributes:
                        reward += important_attributes_counter[attribute]


                except:
                    df_attributes.append(['None', 'None', 'None'])
                    reward = 0.1
                action, _, _ = get_action_reward(interaction)
                df_action.append(action)
                df_reward.append(reward)
                df_state.append(one_hot_encode_state(df_attributes[-1]))
                df_high_state.append(get_state(interaction))

            df['Attribute'] = df_attributes
            df['Reward'] = df_reward
            df['Action'] = df_action
            df['State'] = df_state
            df['High-Level-State'] = df_high_state

            df['User_Index'] = df.index  # Use the existing DataFrame index as User_Index
            df.set_index('User_Index', inplace=True)  # Set 'User_Index' as the index

            # Reset the index before saving to CSV
            df.reset_index(inplace=True)
            # Drop rows where 'Value' is empty
            df = df[df['Value'].notna()]
            # Drop rows where 'Interaction' contains 'typed in answer'
            df = df[~df['Interaction'].str.contains('typed in answer')]
            # Drop rows where 'Interaction' contains 'typed in answer'
            df = df[~df['Interaction'].str.contains('changed ptask ans')]

            df = df[~df['Interaction'].str.contains('main chart changed')]

            df = df[~df['Interaction'].str.contains('clicked on a field')]

            df = df[~df['Interaction'].str.contains('window')]

            df = df[~df['Interaction'].str.contains('study begins')]

            # Fix index
            df.reset_index(drop=True, inplace=True)

            # Reorder columns
            df = df[['User_Index', 'Interaction', 'Value', 'Time', 'Reward', 'Action', 'Attribute', 'State',
                     'High-Level-State']]

            processed_csv_path = os.path.join(processed_interactions_path, csv_filename)
            df.to_csv(processed_csv_path, index=False)  # Use index=False to avoid the "Unnamed: 0" column

    for csv_filename in csv_files:
        if csv_filename.endswith('p4_logs.csv'):
            print("Converting '{}'...".format(csv_filename))

            df = pd.read_csv(os.path.join(processed_interactions_path, csv_filename))
            actions=[]
            for index in range(len(df) - 1):  # Iterate up to the second-to-last row
                current_state = np.array(eval(df['State'][index]))  # Convert string representation to list
                next_state = np.array(eval(df['State'][index + 1]))  # Convert string representation to list
                action=''
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
                # current_action = current_state - next_state
                # df.at[index, 'Action'] = str(list(current_action))  # Convert the NumPy array back to a list and store as a string
            actions.append('same')
            df['Action']=actions

    
        # Save the modified DataFrame
        df.to_csv(os.path.join(processed_interactions_path, csv_filename), index=False)
