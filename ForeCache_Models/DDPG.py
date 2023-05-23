import environment2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import plotting
from collections import Counter
import pandas as pd
import json
import os
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = fname.rstrip('.csv')
    return uname

def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(0)
    return accuracy_per_state

if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    run_experiment(user_list_2D, 'Actor_Critic', 'sampled-hyperparameters-config.json')