import csv

import requests

import environment_vizrec #importing the environment
from openai import OpenAI
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import defaultdict

#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
# Initialize the OpenAI API client
api_key = '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'

class LLM:
    def __init__(self):
        self.base_url = "http://localhost:1234/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.api_key = "lm-studio"
        self.best_action = []

    def generate_prompt(self, interactions):
        context = """
        Context:
        In this task, users are exploring a dataset through a visualization interface. They can modify their current focus by adding or removing column name from the visualization. The goal is to predict the next action based on the user's ALL previous interactions.
        """
        interactions_prompt = ""

        interactions_prompt += f"""Attribute: {interactions['Attribute']}
           Action: {interactions['Action']}"""
        prompt = f"json\n{context}\n\nInteractions:{interactions_prompt}\n\nGiven the Attributes and the action they have taken, predict the next action.ONLY Reply with one action from 4 possible ['same','mofidy','modify-2','modify-3'] you think the user will take next. Avoid extra details"
        return prompt

    def predict_next_action(self, prompt):
        data = {
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": True
        }
        response = requests.post(self.base_url, json=data, headers=self.headers)
        next_action = response.json()['choices'][0]['message']['content']
        return next_action

    def LLMDriver(self, user, env, thres):
        length = len(env.mem_states)
        threshold = int(length * thres)
        attributes= env.mem_states
        actions= env.mem_action



        # Training phase
        correct_predictions = 0
        for i in range(1,threshold):
            interactions = pd.DataFrame({'Attribute': attributes[:i], 'Action': actions[:i]})
            prompt = self.generate_prompt(interactions)
            true_action = actions[i]  # Assuming 'Action' is the key for the true action
            predicted_action = self.predict_next_action(prompt)
            self.best_action.append(predicted_action)
            if predicted_action == true_action:
                correct_predictions += 1
                feedback = "correct"
            else:
                feedback = "incorrect"
            # Provide feedback to the model during training
            # Here you can implement logic to update the model based on feedback
            print(f"Iteration {i}: Predicted: {predicted_action}, True: {true_action}, Feedback: {feedback}")

        # Testing phase
        correct_predictions_test = 0
        for i in range(threshold, length):
            prompt = self.generate_prompt(env[:i])
            predicted_action = self.predict_next_action(prompt)
            true_action = env[i]['Action']  # Assuming 'Action' is the key for the true action
            if predicted_action == true_action:
                correct_predictions_test += 1

        accuracy_train = correct_predictions / threshold
        accuracy_test = correct_predictions_test / (length - threshold)
        return accuracy_train, accuracy_test, self.best_action

# Read interactions from CSV file
def format_split_accuracy(accuracy_dict):
    main_states=['Foraging', 'Navigation', 'Sensemaking']
    accuracy_per_state=[]
    for state in main_states:
        if accuracy_dict[state]:
            accuracy_per_state.append(np.mean(accuracy_dict[state]))
        else:
            accuracy_per_state.append(None) #no data for that state
    return accuracy_per_state

def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname


def run_experiment(user_list, algo, hyperparam_file,task, dataset):
    # Load hyperparameters from JSON file
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(columns=['User', 'Accuracy','Threshold', 'LearningRate', 'Discount','Algorithm','StateAccuracy'])
    y_accu_all=[]
    title=algo
    for u in user_list:
        y_accu = []
        threshold=hyperparams['threshold'][1:]
        user_name = get_user_name(u)
        for thres in threshold:
            env = environment_vizrec.environment_vizrec()
            env.process_data(u, 0)
            obj = LLM()
            test_accuracy, state_accuracy = obj.LLMDriver(user_name, env, thres)
            #accuracy_per_state = format_split_accuracy(state_accuracy)
            y_accu.append(test_accuracy)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame({
                'User': [user_name],
                'Threshold': [thres],
                'LearningRate': [None],
                'Discount': [None],
                'Accuracy': [test_accuracy],
                'StateAccuracy': [0],
                'Algorithm': [title],
                'Reward': [None]
            })], ignore_index=True)
            env.reset(True, False)
        print("User ", user_name, " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=user_name, marker='*')
        y_accu_all.append(np.mean(y_accu))

    print("Random Model Performace: ", "Global Accuracy: ", np.mean(y_accu_all))
    # Save result DataFrame to CSV file
    result_dataframe.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset,task,title), index=False)



if __name__ == "__main__":
    datasets = ['movies']
    tasks =['p1', 'p2', 'p3', 'p4']
    for dataset in datasets:
        for task in tasks:
            env = environment_vizrec.environment_vizrec()
            user_list_name = env.get_user_list(dataset, task)
            run_experiment(user_list_name, 'LLM', 'sampled-hyperparameters-config.json', task, dataset)
            print(f"Done with {dataset} {task}")

