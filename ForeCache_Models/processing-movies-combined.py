import numpy as np
import os
import pandas as pd
from collections import Counter, defaultdict
import csv
from itertools import product
import ast
import json
import itertools


#only get pro22 from 'pro22_ade_p1_logs.csv'
def global_get_user_name(url):
    if 'logs.csv' in url:
        return url.split('_')[0]

if __name__ == '__main__':

    for dataset in ['movies','birdstrikes']:
        if dataset == 'movies':
            processed_interactions_path = './data/zheng/processed_interactions_'
            p1_csv_files = os.listdir(processed_interactions_path + 'p1')
            p2_csv_files = os.listdir(processed_interactions_path + 'p2')
            p3_csv_files = os.listdir(processed_interactions_path + 'p3')
            p4_csv_files = os.listdir(processed_interactions_path + 'p4')
        else:
            processed_interactions_path = './data/zheng/birdstrikes_processed_interactions_'
            p1_csv_files = os.listdir(processed_interactions_path + 'p1')
            p2_csv_files = os.listdir(processed_interactions_path + 'p2')
            p3_csv_files = os.listdir(processed_interactions_path + 'p3')
            p4_csv_files = os.listdir(processed_interactions_path + 'p4')

        # find the user file from these 4 tasks and combine them into a single file, all files have same column headersuse the last value of the Time column to identify the order of merging
        possible_user_names= list(set([global_get_user_name(csv_filename) for csv_filename in p1_csv_files+p2_csv_files+p3_csv_files+p4_csv_files]))
        #drop names not ending in .csv and any None values
        possible_user_names = [name for name in possible_user_names if name is not None]
        possible_user_names = [name for name in possible_user_names if '.csv' not in name]
        print('############################# Total users:##################', len(possible_user_names))
        # fileds :User_Index,Interaction,Value,Time,Reward,Action,Attribute,State,High-Level-State
        # create a combined file per user for all tasks
        for user in possible_user_names:
            user_p1 = [csv_file for csv_file in p1_csv_files if user in csv_file]
            user_p2 = [csv_file for csv_file in p2_csv_files if user in csv_file]
            user_p3 = [csv_file for csv_file in p3_csv_files if user in csv_file]
            user_p4 = [csv_file for csv_file in p4_csv_files if user in csv_file]
            #get the last file for each task
            user_p1 = user_p1[-1]
            user_p2 = user_p2[-1]
            user_p3 = user_p3[-1]
            user_p4 = user_p4[-1]
            #combine the 4 files
            p1_df = pd.read_csv(processed_interactions_path + 'p1/' + user_p1)
            p2_df = pd.read_csv(processed_interactions_path + 'p2/' + user_p2)
            p3_df = pd.read_csv(processed_interactions_path + 'p3/' + user_p3)
            p4_df = pd.read_csv(processed_interactions_path + 'p4/' + user_p4)
            p1_df['Task'] = 'p1'
            p2_df['Task'] = 'p2'
            p3_df['Task'] = 'p3'
            p4_df['Task'] = 'p4'
            combined_df = pd.concat([p1_df, p2_df, p3_df, p4_df])
            #sort this by time increasing lower time first
            combined_df = combined_df.sort_values(by=['Time'])
            combined_df.to_csv(processed_interactions_path + 'all/' + user + '_combined.csv', index=False)
            print('Combined', user)
