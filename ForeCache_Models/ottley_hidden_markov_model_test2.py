import pathlib

import matplotlib.pyplot as plt
import numpy as np
from random import seed, getstate, setstate
from re import search
import time
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from util import flatten_list, lognormpdf
import ast
eps = 2 ** -52  # Defined to match Matlab's default eps
import json
import copy
from collections import Counter

class HMM:
    """
    Hidden Markov Model: An approach for modeling user attention during visual exploratory analysis.

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
        num_particles: An integer representing the number of particles used for diffusion
    """
    def __init__(self, data, continuous_attributes, discrete_attributes, num_particles):

        #same as original implementation
        self.underlying_data = data
        self.underlying_data_w_probability = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.num_particles = num_particles

        # normalize all continuous variables to be between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
#        self.underlying_data[flatten_list(self.continuous_attributes)] = min_max_scaler.fit_transform(self.underlying_data[flatten_list(self.continuous_attributes)])

        # initiate the pi vector to 0.5 for every attribute; and sigma for pi is 0.1; these values are arbitrary at the moment
        # self.bias_vector = 0.5 * np.ones(len(continuous_attributes) + len(discrete_attributes))
        self.bias_sigma = 0.1

        # initiate the sigma vector for every continuous attribute; set it to underlying data's standard deviation
        #self.continuous_sigma = {attr: 0.25*self.underlying_data[attr].std() for attr in flatten_list(self.continuous_attributes)}

        # initiate the p vector for discrete attributes; p is the probability the user's attention flips to some other value of a given attribute
        self.discrete_p = {attr: 1/len(self.underlying_data[attr].unique()) for attr in self.discrete_attributes}

        # initiate particles randomly; particles are of form [c1, c2, c3, ..., d1, d2, d3, ..., pi1,pi2, pi3,...]
        # where c is a continuous attr, d is a discrete attribute, and pi is a bias value
        attribute_names = [x if type(x) == str else '___'.join(x) for x in continuous_attributes + discrete_attributes]
        self.bias_column_names = [f'bias_{x}' for x in attribute_names]
        self.bias_over_time = pd.DataFrame(columns=self.bias_column_names)

        self.particles = pd.DataFrame(columns=flatten_list(continuous_attributes) + discrete_attributes + self.bias_column_names)
        # for attr in flatten_list(continuous_attributes):
        #     self.particles[attr] = np.random.rand(self.num_particles)
        for attr in self.bias_column_names:
            self.particles[attr] = np.random.rand(self.num_particles)
        for attr in discrete_attributes:
            unique_values = self.underlying_data[attr].unique()
            proportions = self.underlying_data[attr].value_counts(normalize=True)[unique_values]
            self.particles[attr] = np.random.choice(unique_values, p=proportions, size=self.num_particles)

    def update(self, observation_index):
        """
        Update the HMM model.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data 
        """
        # defuse particles
        self.defuse_particles()

        # get "click probabilities"; this is a |particles| x |underlying_data| matrix
        # element i, j is the probability of observing particle i given that data point j was our last observation.
        probability_matrix = self.get_probability_matrix()

        # weight particles by evidence and resample particles according to the weights
        weights = probability_matrix[:, observation_index]
        resampled_indices = np.random.choice(self.num_particles, self.num_particles, p=weights/np.sum(weights))
        self.particles = self.particles.iloc[resampled_indices]

        # record click probabilities
        data_point_probabilities = probability_matrix.mean(axis=0)
        self.underlying_data_w_probability['probability'] = data_point_probabilities

        self.bias_over_time = pd.concat([self.bias_over_time,self.particles[self.bias_column_names].mean()], ignore_index=True)

    def defuse_particles(self):
        """
        Diffusion of particles on all attributes
        """

        # diffusion on continuous attributes
        # for attr in flatten_list(self.continuous_attributes):
        #     self.particles[attr] = self.particles[attr] + self.continuous_sigma[attr] * np.random.normal(size=self.num_particles)

        # diffusion on bias variables
        for attr in self.bias_column_names:
            self.particles[attr] = self.particles[attr] + self.bias_sigma * np.random.normal(size=self.num_particles)

        # diffusion on types ("discrete diffusion" as defined in the paper)
        for attr in self.discrete_attributes:
            flip_indices = np.random.random(self.num_particles) < self.discrete_p[attr]
            unique_values = self.underlying_data[attr].unique()
            proportions = self.underlying_data[attr].value_counts(normalize=True)[unique_values]
            self.particles.loc[flip_indices, attr] = np.random.choice(unique_values, size=np.count_nonzero(flip_indices), p=proportions)

        # clip values to boundary
        #self.particles[flatten_list(self.continuous_attributes)] = self.particles[flatten_list(self.continuous_attributes)].clip(lower=eps, upper=1-eps)
        self.particles[self.bias_column_names] = self.particles[self.bias_column_names].clip(lower=eps, upper=1-eps)

    def get_probability_matrix(self):
        """
        This function returns a |particles| x |data_points| matrix of probabilities
        Element i, j is the probability of particle i given that data point j was our last observation.
        
        Returns:
            A numpy matrix
        """
        particles = self.particles
        data_points = self.underlying_data
        probability_matrix = np.zeros((len(particles), len(data_points)))

        # # for continuous attributes, find the probabilities
        # for attr in self.continuous_attributes:
        #     # handling cases where multiple continuous attributes come together with one bias (e.g. lat & lng coming together as location)
        #     attr_name = attr
        #     attr_list = attr
        #     if type(attr) != str:
        #         attr_name = '___'.join(attr)
        #     else:
        #         attr_list = [attr]
        #
        #     attr_log_pdf = np.zeros((len(particles), len(data_points)))
        #     for continuous_attr in attr_list:
        #         attr_log_pdf += lognormpdf(particles[continuous_attr].to_numpy().reshape(-1, 1), data_points[continuous_attr].to_numpy(), self.continuous_sigma[continuous_attr])
        #
        #     # shift for numerical stability purposes
        #     attr_log_pdf -= np.max(attr_log_pdf, axis=1)[..., np.newaxis]
        #     # print(attr_log_pdf)
        #
        #     # exponential to get pdf from logpdf
        #     attr_log_pdf = np.exp(attr_log_pdf)
        #
        #     # normalize to get pmf
        #     attr_log_pdf = attr_log_pdf / np.sum(attr_log_pdf, axis=1)[..., np.newaxis]
        #
        #     # add the weighted probabilities according to particle bias value
        #     bias = particles[f'bias_{attr_name}'].to_numpy()
        #     probability_matrix += bias.reshape(-1, 1) * attr_log_pdf

        # for discrete attributes, find probability
        for attr in self.discrete_attributes:
            attr_probability = particles[attr].to_numpy().reshape(-1, 1) == data_points[attr].to_numpy()
            attr_probability = attr_probability / attr_probability.sum(axis=1)[..., np.newaxis]

            # add the weighted probabilities according to particle bias value
            bias = particles[f'bias_{attr}'].to_numpy()
            probability_matrix += bias.reshape(-1, 1) * attr_probability

        return probability_matrix

    def predict(self):
        """
        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        return self.underlying_data_w_probability['probability']

    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return self.bias_over_time

def process_actions(current_state,next_state):
       # print("Converting '{}'...".format(csv_filename))

       # Count occurrences of each element in current_state and next_state
        current_state_count = Counter(current_state)
        next_state_count = Counter(next_state)

        # Get the elements that are in next_state but not in current_state
        difference_elements = list((next_state_count - current_state_count).elements())

        # Get the number of different elements
        num_different_elements = len(difference_elements)

        if num_different_elements == 0:
            action = 'same'
        else:
            action = f'modify-{num_different_elements}'
            print('Reset')

        return action

def get_action_from_next_point(original_data,next_points,last_point):
    real_actions=[]
    #get the action for the last_point
    last_point = last_point
    selected_rows = original_data.loc[original_data['id'] == last_point]
    last_attribute = selected_rows.iloc[-1,:].values[1:4]
    for next_point in next_points:
        # find that next_point in underlying data and get the last column value
        selected_rows = original_data.loc[original_data['id'] == next_point]
        attribute_for_point = selected_rows.iloc[-1,:].values[1:4]
        action_for_point = process_actions(last_attribute,attribute_for_point)
        real_actions.append(action_for_point)
    return real_actions[0]


def plot_results(output_file_path):
    ks = [1, 5]
    zeng_map_results = pd.read_pickle(output_file_path)
    df_temp = zeng_map_results[[f'ncp-{k}' for k in ks]]
    err = df_temp.std() / np.sqrt(len(df_temp))
    df_temp.mean().plot.bar(yerr=err, color='#d95f02', alpha=0.5,
                            title=f'Aggregate Next Action Prediction for Movies Data')
    plt.show()




if __name__ == '__main__':
    hyperparam_file='sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    session=1
    dataset = 'movies'
    underlying_data_paths = {
        'movies': './data/zheng/combinations.csv'   }

    user_interaction_data_paths = {
        'movies': './data/zheng/competing_movies_interactions.csv'    }

    continuous_attributes = {
        'movies': []
    }

    discrete_attributes = {
        'movies': ['attribute1']
    }

    output_file_path = './output/movies/movies_results_test_hmm.pkl'


    underlying_data = pd.read_csv(underlying_data_paths[dataset])
    original_data=pd.read_csv(underlying_data_paths[dataset])
    underlying_data.set_index('id', drop=True, inplace=True)



    interaction_data = pd.read_csv(user_interaction_data_paths[dataset])[:1]
    interaction_data['interaction_session'] = interaction_data.apply(
        lambda row: ast.literal_eval(row.interaction_session), axis=1)

    hmm = HMM(underlying_data,
              continuous_attributes[dataset].copy(),
              discrete_attributes[dataset].copy(),
              1000)

    # Not necessary to run if we already have results file for HMM
    # Running HMM through all user interaction sessions and saving results in file
    hmm_results = pd.DataFrame()
    ks= [1]
    all_threshold = hyperparams['threshold']

    # Create result DataFrame with columns for relevant statistics
    result_dataframe = pd.DataFrame(
        columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])
    results={}
    for participant_index, row in interaction_data.iterrows():
        print(f'Processing user {row.user}')
        user_data= interaction_data.iloc[participant_index].interaction_session
        results ['participant_id']= row.user
        length = len(user_data)
        for thres in all_threshold:
            threshold = int(length * thres)
            print("threshold", threshold, "length", length - 1)
            #uderlying_data is all possible states and actions
            hmm = HMM(underlying_data, continuous_attributes['movies'],discrete_attributes['movies'], 1000)
            predicted = pd.DataFrame()
            rank_predicted = []

            #training the model
            for k in range(threshold + 1):
                interaction = interaction_data.iloc[participant_index].interaction_session[k]
                hmm.update(interaction)


            #testing the model
            for i in tqdm(range(threshold+1 , len(interaction_data.iloc[participant_index].interaction_session))):
                interaction = interaction_data.iloc[participant_index].interaction_session[i]
                hmm.update(interaction)
                last_point = interaction_data.iloc[participant_index].interaction_session[i]
                if i < len(interaction_data.iloc[participant_index].interaction_session) - 1:
                    probability_of_next_point = hmm.predict()
                    next_point = interaction_data.iloc[participant_index].interaction_session[i + 1]

                    predicted_next_dict = {}
                    for k in ks:
                        print('Testing for k:', k)
                        predicted_next_point=probability_of_next_point.nlargest(k).index.values
                        #get the action for the next point/s since when k>1 we have multiple next points
                        action_predicted = get_action_from_next_point(original_data,predicted_next_point,last_point)
                        #get the actual action from the actual next_point
                        action_true= get_action_from_next_point(original_data,[next_point],last_point)
                        predicted_next_dict[k] = (action_true == action_predicted)
                    predicted = pd.concat([predicted,pd.DataFrame(predicted_next_dict, index=[0])], ignore_index=True)
                    sorted_prob = probability_of_next_point.sort_values(ascending=False)
                    rank, = np.where(sorted_prob.index.values == next_point)
                    rank_predicted.append(rank[0] + 1)
                    last_point = next_point

            # ncp = predicted.sum() / len(predicted)
            # # for col in ncp.index:
            # #     results[f'ncp-{col}'] = ncp[col]
            results['user'] = row.user
            results['threshold'] = thres
            results[f'ncp-{1}'] = predicted[1].sum()/len(predicted)
            # results[f'ncp-{5}'] = predicted[5].sum()/len(predicted)

            # we dont care about bias

            #bias = hmm.get_attribute_bias()
            # for col in bias.columns:
            #     results[f'bias-{col}'] = bias[col].to_numpy()
            #
            # results['bias-mixed'] = results['bias-bias_attribute1___attribute2___attribute3'] * results['bias-bias_action']
            results['rank'] = rank_predicted
            hmm_results = pd.concat([hmm_results,pd.DataFrame(results)], ignore_index=True)

    hmm_results.to_pickle(output_file_path)
    plot_results(output_file_path)