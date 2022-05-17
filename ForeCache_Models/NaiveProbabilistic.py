import numpy as np
import pandas as pd
from collections import defaultdict
import glob

#for plots
import matplotlib.pyplot as plt
from textwrap import wrap


class NaiveProbabilistic():

    def __init__(self):
        self.user_list = glob.glob('taskname_ndsi-2d-task_*')
        self.valid_actions = [0,1]
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0

    def process_data(self, filename, thres):
        df = pd.read_csv(filename)
        self.prev_state = None
        cnt_inter = 0
        for index, row in df.iterrows():
            cur_state = row['State']
            if cur_state not in ('Foraging', 'Navigation', 'Sensemaking'):
                continue
            if self.prev_state == cur_state:
                action = 0
            else:
                action = 1
            self.mem_states.append(cur_state)
            self.mem_reward.append(row['NDSI'])
            self.mem_action.append(action)
            cnt_inter += 1
            self.prev_state = cur_state
        self.threshold = int(cnt_inter * thres)
        print("{} {}\n".format(len(self.mem_states), self.threshold))


    def probabilistic_learning(self):
        self.Naive_Q = defaultdict(lambda: np.zeros(len(self.valid_actions)))
        train_states=self.mem_states[:self.threshold]
        train_actions=self.mem_action[:self.threshold]
        for i in range(len(train_states)) :
                self.Naive_Q[train_states[i]][train_actions[i]]+=1
       # Dict = dict({'Sensemakingchange':  Naive_Q['Sensemaking'][1]/len(Naive_Q['Sensemaking']), 2: 'For', 3:'Geeks'})
        return self.Naive_Q

    def test(self):
        predicted_actions=[]
        acc=0
        test_states=self.mem_states[self.threshold:]
        test_actions=self.mem_action[self.threshold:]
        for i in range(len(test_states)):
            predicted_actions.append(np.argmax(self.Naive_Q[test_states[i]]))
            if test_actions[i]==predicted_actions[i]:
                acc+=1
        return acc/len(test_actions) #final accuracy

if __name__ == "__main__":
    accuracies = []
    plot_list=[]
    user_index=[]
    obj=NaiveProbabilistic()
    users = obj.user_list
    for i in range(len(users)):
        plot_list.append(i)
        user_index.append(users[i][29:-4])
        thres = 0.7 #the percent of interactions Naive Probabilistic will be trained on
        print('########For user#############',users[i])
        obj.process_data(users[i], thres)
        obj.probabilistic_learning()
        accuracy=obj.test()
        accuracies.append(accuracy)

    #plotting
    plt.figure(figsize=[10,10])
    plt.plot(plot_list,accuracies, '-bo', label='Naive Probabilistic learning Average Test Accuracy for Users 1-20')
    # plot_list = [l[35:] for l in plot_list]
    plt.xticks(plot_list, rotation='vertical')
    plt.margins(0.002)
    plt.xlabel("Users 1 - 20")
    plt.ylabel("Test Accuracy on action prediction")
    plt.legend(loc='upper right')
    plt.show()