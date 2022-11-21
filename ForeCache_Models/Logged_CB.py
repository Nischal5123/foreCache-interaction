import vowpalwabbit
import numpy as np
import pandas as pd
import random
from vowpalwabbit.dftovw import DFtoVW
from vowpalwabbit import Workspace
import os
import glob
from sklearn.utils import shuffle
pd.set_option('mode.chained_assignment', None)


class ContextualBandit:
    def __init__(self,flag):
        self.valid_actions=[2,1]
        self.user_list_2D = np.sort(glob.glob('data/NDSI-2D/taskname_ndsi-2d-task_*'))
        self.policy_flag=str(flag)

    def process_data(self, user_location):
        df = pd.read_csv(user_location)
        user = user_location.lstrip('data/NDSI-2D\\taskname_ndsi-2d-task_')
        print("#####################   ", user, "    #######################")

        for i in range(len(df)):
            if df.loc[i, "Action"] == 'zoom':
                df.loc[i, "Action"] = self.valid_actions[0]
            else:
                df.loc[i, "Action"] = self.valid_actions[1]
        return df

    def train_test_split(self, df, thres):
        #df=shuffle(df)
        train_df = df[:round(thres * len(df))]
        test_df = df[round(thres * len(df)):]
        train_df["index"] = range(1, len(train_df) + 1)
        train_df = train_df.set_index("index")
        test_df["index"] = range(1, len(test_df) + 1)
        test_df = test_df.set_index("index")
        return train_df, test_df

    def cost_func(self,data):
        return (1-data['NDSI'])


    def train(self,data):
        vw = vowpalwabbit.Workspace(self.policy_flag,quiet=False)

        # use the learn method to train the vw model, train model row by row using a loop
        for i in data.index:
            ## provide data to cb in requested format
            action = data.loc[i, "Action"]
            cost = self.cost_func(data.loc[i])
            probability = data.loc[i, "StateActionProbab"]
            feature1 = data.loc[i, "Most_frequent_region"]
            feature2 = data.loc[i, "Subtask_ROI"]

            ## do the actual learning
            vw.learn(
                str(action)
                + ":"
                + str(cost)
                + ":"
                + str(probability)
                + " | "
                + str(feature1)
                + " "
                + str(feature2)
            )
        return vw

    def sample_prediction(action_probs):
        "return the index of the selected action, and the probability of that action"
        [selected_index] = random.choices(range(len(action_probs)), weights=action_probs)
        return selected_index, action_probs[selected_index]

    def test(self,data,model):
        pred=[]
        correct=0
        size=len(data)

        for j in data.index:
            feature1 = data.loc[j, "Most_frequent_region"]
            feature2 = data.loc[j, "Subtask_ROI"]
            format = "| " + str(feature1) + " " + str(feature2)
            choice = model.predict(format)

            print(j, choice, feature1)

            if hasattr(choice, "__len__"):
                chosen_action_index, prob = self.sample_prediction(choice)
                prediction= self.valid_actions[chosen_action_index]

            else:
                prediction=choice
            correct += (prediction == data.loc[j, "Action"])
            pred.append(prediction)

        print("accuracy: ",correct/size)
        print(pred)
        return pred





if __name__ == "__main__":
    cb = ContextualBandit("--cb 2 --cb_type ips")
    users = cb.user_list_2D
    user_list_experienced = np.array(
        ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_3abeecbe-327a-441e-be2a-0dd3763c1d45.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
    for u in user_list_experienced:
        cb = ContextualBandit("--cb 2 -q :: --leave_duplicate_interactions --cb_type ips")
        data=cb.process_data(u)
        train_df, test_df=cb.train_test_split(data,0.80)
        model=cb.train(train_df)
        predictions=cb.test(test_df,model)
