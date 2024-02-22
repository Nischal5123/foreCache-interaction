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


if __name__ == '__main__':
    original_json_path = './data/zheng/logs'
    processed_interactions_path = './data/zheng/birdstrikes_processed_csv'
    all_csv_files = os.listdir(original_json_path)
    csv_files = [file for file in all_csv_files if file.split('_')[1].startswith('b')]
    # if file ends with _logs.json its a interaction log file
    json_interactions_files = [file for file in csv_files if file.endswith('_logs.json')]
    json_to_csv(json_interactions_files, original_json_path, processed_interactions_path)
