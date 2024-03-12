import os
import json
import csv
import os
import pandas as pd
def json_to_csv(json_interactions_files, original_json_path, processed_interactions_path):
    for f in json_interactions_files:
        with open(os.path.join(original_json_path, f)) as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            #df.to_csv(os.path.join(processed_interactions_path, f.replace('.json' , '.csv'), index=False)
            df.to_csv(os.path.join(processed_interactions_path, f.split('_logs.json')[0] + '_logs.csv'), index=False)
        print(f'Processed {f} to csv')
    print(' ###### All files processed to csv ######')


def json_to_csv_bookmark(json_bookmark_files, original_bookmark_json_path, processed_bookmarks_interactions_path):
    for f in json_bookmark_files:
        with open(os.path.join(original_bookmark_json_path, f)) as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(processed_bookmarks_interactions_path, f.split('_logs.json')[0] + '.csv'), index=False)
        print(f'Processed {f} to csv')
    print(' ###### All files processed to csv ######')

def verify_file_creation(file_path, dataset):
    verify_files = os.listdir(file_path)
    print(verify_files)
    print(len(verify_files), 'files processed to csv')

    # 1) asset each file ends in logs.csv
    assert all([file.endswith('_logs.csv') for file in verify_files])

    # 2) assert each file starts with a or b
    if dataset == 'birdstrikes':
        for file in verify_files:
            assert 'b' in file.split('_')[1]
            assert 'a' not in file.split('_')[1]
    else:
        for file in verify_files:
            assert 'a' in file.split('_')[1]
            assert 'b' not in file.split('_')[1]

    # 3) assert there are 36 unique user
    unique_users = set([file.split('_')[0] for file in verify_files])
    print(len(unique_users), 'unique users')
    #assert unique users is equal to 36
    assert len(unique_users) == 36

def global_remove_invalid_rows(user_path, csv_filename):

    df = pd.read_csv(os.path.join(user_path, csv_filename))
    # Drop rows where 'Interaction' contains 'typed in answer'
    df = df[~df['Interaction'].str.contains('typed in answer')]
    # Drop rows where 'Interaction' contains 'changed ptask ans'
    df = df[~df['Interaction'].str.contains('changed ptask ans')]
    df = df[~df['Interaction'].str.contains('window')]
    df = df[~df['Interaction'].str.contains('study begins')]

    # remove all rows with null in 'Value' column
    df = df.dropna(subset=['Value'])
    df = df[~df['Value'].str.contains('remove undefined')]

    df = df.reset_index(drop=True)

    prev_time_hover = 0
    prev_time_scroll = 0

    for index, row in df.iterrows():
        interaction = row['Interaction'] = row['Interaction']

        if 'bookmark' in interaction or 'main chart changed' in interaction or 'clicked on a field' in interaction:
            pass

        elif 'mouse' in interaction:
            time_period = (row['Time'] - prev_time_hover) / 1000
            if time_period < 0.5:
                # drop the row
                df.drop(index, inplace=True)
                prev_time_hover = row['Time']

        elif 'scroll' in interaction:
            time_period = (row['Time'] - prev_time_scroll) / 1000
            if time_period < 0.5:
                # drop the row
                df.drop(index, inplace=True)
                prev_time_scroll = row['Time']

        else:
            pass

    df.to_csv(os.path.join(user_path, csv_filename), index=False)



if __name__ == '__main__':
    original_json_path = './data/zheng/logs'
    dataset= 'birdstrikes'
    if dataset == 'birdstrikes':
        processed_interactions_path = './data/zheng/birdstrikes_processed_csv'
        all_csv_files = os.listdir(original_json_path)
        csv_files = [file for file in all_csv_files if file.split('_')[1].startswith('b')]
    else:
        processed_interactions_path = './data/zheng/processed_csv'
        all_csv_files = os.listdir(original_json_path)
        csv_files = [file for file in all_csv_files if file.split('_')[1].startswith('a')]
    # if file ends with _logs.json its a interaction log file
    json_interactions_files = [file for file in csv_files if file.endswith('_logs.json')]
    json_to_csv(json_interactions_files, original_json_path, processed_interactions_path)

    csv_files = os.listdir(processed_interactions_path)
    current_csv_files = []
    for csv_filename in csv_files:
        end = '_logs.csv'
        if csv_filename.endswith(end):
            global_remove_invalid_rows(processed_interactions_path, csv_filename)

    print ('All interaction files processed to csv')

    verify_file_creation(processed_interactions_path, dataset)


