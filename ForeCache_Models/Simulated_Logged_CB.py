import vowpalwabbit
import numpy as np
import pandas as pd
import random
import json
import matplotlib as plt
import os
import glob
from sklearn.utils import shuffle
pd.set_option('mode.chained_assignment', None)


class ContextualBandit:
    def __init__(self,flag):
        self.valid_actions=['Sensemaking','Foraging','Navigation']
        self.valid_contexts=['NorthernRockiesPlains','Northeast','NorthWest','SouthWest','Other']
        self.user_list_2D = np.sort(glob.glob('data/NDSI-2D/taskname_ndsi-2d-task_*'))
        self.policy_flag=str(flag)

    def process_data(self, user_location):
        df = pd.read_csv(user_location)
        user = user_location.lstrip('data/NDSI-2D\\taskname_ndsi-2d-task_')
        print("#####################   ", user, "    #######################")

        # for i in range(len(df)):
        #     if df.loc[i, "Action"] == 'zoom':
        #         df.loc[i, "Action"] = self.valid_actions[0]
        #     else:
        #         df.loc[i, "Action"] = self.valid_actions[1]
        for i in range(len(df)):
            if df.loc[i, "State"] == self.valid_actions[0]:
                df.loc[i, "State"] = 1
            elif df.loc[i, "State"] == self.valid_actions[1]:
                df.loc[i, "State"] = 2
            else:
                df.loc[i, "State"] = 3
            if df.loc[i, "Most_frequent_region"] in ('NorthernRockiesPlains','Northeast','NorthWest','SouthWest'):
                df.loc[i, "Most_frequent_region"]=df.loc[i, "Most_frequent_region"]
            else:
                df.loc[i, "Most_frequent_region"]= 'Other'
        return df

    def choose_context(self):
        return random.choice(self.valid_contexts)

    def to_vw_example_format(self,context, cats_label=None):
        example_dict = {}
        if cats_label is not None:
            chosen_state, cost, pdf_value = cats_label
            example_dict["_label_ca"] = {
                "action": chosen_state,
                "cost": cost,
                "pdf_value": pdf_value,
            }
        example_dict["c"] = {
            "region={}".format(context["region"]): 1
        }
        return json.dumps(example_dict)

    def predict_state(self,vw, context):
        vw_text_example = self.to_vw_example_format(context)
        return vw.predict(vw_text_example)

    def run_simulation(self,
            vw,
            num_iterations,
            states,
            regions,
            cost_func,
            df,
            do_learn=True,
    ):

        reward_rate = []
        hits = 0
        cost_sum = 0.0

        for i in range(1, num_iterations + 1):
            for j in range(len(df)):

                data=data.loc[j]
                region = self.choose_context()
                # 3. Pass context to vw to get a temperature
                context = {"region": region}
                state, pdf_value = self.predict_state(vw, context)

                # 4. Get cost of the action we chose
                cost = cost_func(data)
                cost_sum += cost

                if do_learn:
                    # 5. Inform VW of what happened so we can learn from it
                    txt_ex = self.to_vw_example_format(
                        context, cats_label=(state, cost, pdf_value)
                    )
                    vw_format = vw.parse(txt_ex, vowpalwabbit.LabelType.CONTINUOUS)
                    # 6. Learn
                    vw.learn(vw_format)
                    # 7. Let VW know you're done with these objects
                    vw.finish_example(vw_format)

                # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
                reward_rate.append(-1 * cost_sum / i)

        return reward_rate, hits

    def plot_reward_rate(self,num_iterations, reward_rate, title):
        plt.show()
        plt.plot(range(1, num_iterations + 1), reward_rate)
        plt.xlabel("num_iterations", fontsize=14)
        plt.ylabel("reward rate", fontsize=14)
        plt.title(title)
        plt.ylim([0, 1])


    def cost_func(self,data):
        return (1-data['NDSI'])








if __name__ == "__main__":
    num_iterations = 5000
    num_actions = 3
    cb = ContextualBandit("--cb 3 -q :: --cb_type ips")
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
    for u in [user_list_experienced[0]]:
        cb = ContextualBandit("--cb 2 -q :: --leave_duplicate_interactions --cb_type ips")
        data=cb.process_data(u)
        train_df, test_df=cb.train_test_split(data,0.90)
        model=cb.train(train_df)
        predictions=cb.test(test_df,model)
