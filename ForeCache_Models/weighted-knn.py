import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.metrics import log_loss, pairwise_distances
from scipy.optimize import minimize, Bounds
from scipy.sparse import csc_matrix
from util import flatten_list, eq_dist_function
import json
import environment_vizrec
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.preprocessing import normalize



class WeightedKNN:
    """
    k-Nearest Neighbors: A method for learning users' data interest by observing interaction. This approach assumes that proximity drives a user's exploration patterns.

    Parameters:
        data: A Pandas Dataframe of the underlying data
        continuous_attributes: An array of all continuous attributes in the underlying data
        discrete_attributes: An array of all discrete attributes in the underlying data
        weight_matrix_dir_path: Path to the weight matrix, default is none
        alpha: An array that holds the prior probability of relevance
        k: An integer that represents the number of neighbors used
        num_restarts: An integer that represents the number of restarts
    """
    def __init__(self, data, continuous_attributes, discrete_attributes, weight_matrix_dir_path=None, alpha=[0.9, 0.1], k=50, num_restarts=50):
        self.underlying_data = data
        self.underlying_data_w_probability = data.copy()
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.alpha = alpha  # prior probability of relevance;
        self.q_vector = np.ones(len(continuous_attributes) + len(discrete_attributes))
        self.weights = {}
        self.k = min(k, len(data))
        self.bias_over_time = pd.DataFrame()

        self.num_restarts = num_restarts  # for optimizing q
        self.bounds = Bounds(0, 1)  # for optimizing q

        self.interaction_indices = np.array([], dtype=int)

        # check if the neighbors' matrix exists
        if weight_matrix_dir_path is None:
            # # if not compute neighborhood matrices
            # for i, attr in enumerate(self.continuous_attributes):
            #     attribute_name = attr
            #     if type(attr) == list:
            #         attribute_name = '___'.join(attr)
            #     else:
            #         self.continuous_attributes[i] = [attr]
            #         attr = [attr]
            #     print(f'Computing neighborhood matrix for {attribute_name}')
            #     knn_indices = np.zeros((len(self.underlying_data), self.k))
            #     delta = min(3000, len(self.underlying_data))
            #     for i in range(0, len(self.underlying_data), delta):
            #         distance = pairwise_distances(self.underlying_data.iloc[i:i+delta][attr], self.underlying_data[attr])
            #         for ri in range(delta):
            #             for cj in range(i, i + delta):
            #                 if ri + i == cj:  # if on diagonal
            #                     distance[ri, cj] = np.inf
            #         for j in range(delta):
            #             sorted_indices = np.argsort(distance[j, :])[:k]
            #             knn_indices[i + j, :] = sorted_indices
            #     rows = np.array([])
            #     cols = np.array([])
            #     for i in range(len(knn_indices)):
            #         rows = np.append(rows, np.zeros(len(knn_indices[i])) + i)
            #         cols = np.append(cols, knn_indices[i])
            #     data = np.ones(len(rows)).astype(int)
            #     knn_weights = csc_matrix((data, (rows.astype(int), cols.astype(int))))
            #     self.weights[attribute_name] = knn_weights

            for attr in self.discrete_attributes:
                print(f'Computing neighborhood matrix for {attr}')
                enc = OneHotEncoder()
                data = enc.fit_transform(self.underlying_data[attr].to_numpy().reshape(-1, 1))
                knn_indices = np.zeros((len(self.underlying_data), self.k))
                # knn_indices = {}
                delta = min(3000, len(self.underlying_data))
                for i in range(0, len(self.underlying_data), delta):
                    distance = pairwise_distances(data[i:i + delta, :], data[:, :])
                    for ri in range(delta):
                        for cj in range(i, i + delta):
                            if ri + i == cj:  # if on diagonal
                                distance[ri, cj] = np.inf
                    for j in range(delta):
                        sorted_indices = np.argsort(distance[j, :])[:k]
                        knn_indices[i + j, :] = sorted_indices
                        # indices = np.where(distance[j, :] == 0)[0]
                        # knn_indices[i+j] = indices
                rows = np.array([])
                cols = np.array([])
                for i in range(len(knn_indices)):
                    rows = np.append(rows, np.zeros(len(knn_indices[i])) + i)
                    cols = np.append(cols, knn_indices[i])
                data = np.ones(len(rows)).astype(int)
                knn_weights = csc_matrix((data, (rows.astype(int), cols.astype(int))))
                self.weights[attr] = knn_weights
        else:
            # if so, load the weight matrices
            None

    def update(self, observation_index):
        """
        Update the k-NN model.

        Parameters:
            observation_index: An integer that represents an id of a data point in the underlying data
        """
        self.interaction_indices = np.append(self.interaction_indices, observation_index)
        train_ind = self.interaction_indices
        observed_labels = np.ones(len(train_ind))

        def get_loss(q):
            old_q = self.q_vector
            self.q_vector = q
            probs = self.predict()[train_ind]
            self.q = old_q
            return log_loss(observed_labels, probs, labels=[0, 1])

        min_val = float('inf')
        best_q = None

        for q0 in np.random.uniform(size=(self.num_restarts, len(self.q_vector))):
            res = minimize(
                get_loss, x0=q0, bounds=self.bounds, method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun
                best_q = res.x
        self.q_vector = best_q


        biases = {k: self.q_vector[i] for i, k in enumerate(self.weights)}
        self.bias_over_time = pd.concat([self.bias_over_time, pd.DataFrame(biases,index=[0])], ignore_index=True)

    def predict(self):
        """
        Prediction step where the probability of each point being the next interaction is calculated

        Returns:
            A Pandas Dataframe of the underlying data with probabilities that represent certainity that the point is the next interaction point
        """
        train_ind = self.interaction_indices
        test_ind = np.arange(len(self.underlying_data))
        observed_labels = np.ones(len(train_ind))
        probs = {}
        total_prob = np.zeros(len(test_ind))
        for i, m in enumerate(self.weights):
            weights = self.weights[m]
            attr_probs = np.empty((len(self.underlying_data), 2))
            pos_ind = (observed_labels == 1)
            masks = [~pos_ind, pos_ind]

            for class_ in range(2):
                tmp_train_ind = train_ind[masks[class_]]
                attr_probs[:, class_] = self.alpha[class_] + (
                    weights[:, tmp_train_ind][test_ind].sum(axis=1).flatten()
                )

            attr_probs = normalize(attr_probs, axis=1, norm='l1')[:, 1]
            total_prob += self.q_vector[i] * attr_probs
            probs[m] = attr_probs

        self.underlying_data_w_probability['probability'] = total_prob / self.q_vector.sum()

        return self.underlying_data_w_probability['probability']

    def get_attribute_bias(self):
        """
        Retrieves the calcluated biases of each attribute

        Returns:
            A Pandas Dataframe of biases
        """
        return self.bias_over_time



def get_user_name(url):
    parts = url.split('/')
    fname = parts[-1]
    uname = fname.rstrip('_log.csv')
    return uname

def get_action_from_next_point(original_data,next_points):
    real_actions=[]
    for next_point in next_points:
        # find that next_point in underlying data and get the last column value
        selected_rows = original_data.loc[original_data['id'] == next_point]
        action_for_point = selected_rows.iloc[:, -1].values[0]
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


def create_underlying_and_user_data(user_interactions_path,username):
    df_id=[]
    df_state=[]
    user_data_df=pd.read_csv(user_interactions_path)

    for i in range(len(user_data_df)):
        df_id.append(i)
        df_state.append(user_data_df['State'][i])
    underlying_data = pd.DataFrame({'id': df_id, 'state': df_state})
    user_interaction_data = pd.DataFrame({'user': [username], 'interaction_session': [df_id]})
    print("Underlying data created")
    return underlying_data, user_interaction_data

def user_list(task, dataset):
    env = environment_vizrec.environment_vizrec()
    csv_files=env.get_user_list(dataset,task)
    current_csv_files = []
    for csv_filename in csv_files:
        end = task + '_logs.csv'
        if csv_filename.endswith(end):
            current_csv_files.append(csv_filename)

    return current_csv_files

def user_location(task, dataset):
    env = environment_vizrec.environment_vizrec()
    location=env.get_user_location(dataset,task)
    return location

if __name__ == '__main__':
    hyperparam_file='sampled-hyperparameters-config.json'
    with open(hyperparam_file) as f:
        hyperparams = json.load(f)

    for dataset in ['birdstrikes']:
        for task in ['p1', 'p2', 'p3', 'p4']:
            all_user_files= user_list(task, dataset)


            # Not necessary to run if we already have results file for HMM
            # Running HMM through all user interaction sessions and saving results in file
            hmm_results = pd.DataFrame()
            ks= [1]
            all_threshold = hyperparams['threshold']

            # Create result DataFrame with columns for relevant statistics
            result_dataframe = pd.DataFrame(
                columns=['User', 'Accuracy', 'Threshold', 'LearningRate', 'Discount', 'Algorithm', 'StateAccuracy'])
            results={}
            for index, value in enumerate(all_user_files):
                user_name = get_user_name(value)
                participant_index = 0
                print(f'Processing user {user_name}')
                underlying_data , interaction_data = create_underlying_and_user_data(value,user_name)

                user_data= interaction_data.iloc[participant_index].interaction_session
                results ['participant_id']= user_name
                length = len(user_data)
                for thres in all_threshold:
                    threshold = int(length * thres)
                    print("threshold", threshold, "length", length - 1)
                    #uderlying_data is all possible states and actions
                    hmm = WeightedKNN(underlying_data, [],['state'])
                    predicted = pd.DataFrame()
                    rank_predicted = []

                    #training the model
                    for k in range(threshold + 1):
                        interaction = interaction_data.iloc[participant_index].interaction_session[k]
                        hmm.update(interaction)


                    #testing the model
                    for i in (range(threshold+1 , len(interaction_data.iloc[participant_index].interaction_session))):
                        interaction = interaction_data.iloc[participant_index].interaction_session[i]
                        hmm.update(interaction)

                        if i < len(interaction_data.iloc[participant_index].interaction_session) - 1:
                            probability_of_next_point = hmm.predict()
                            next_point = interaction_data.iloc[participant_index].interaction_session[i + 1]
                            predicted_next_dict = {}
                            for k in ks:
                                print('Testing for k:', k)
                                predicted_next_point=probability_of_next_point.nlargest(k).index.values
                                #get the action for the next point/s since when k>1 we have multiple next points
                                action_predicted = get_action_from_next_point(underlying_data.copy(),predicted_next_point)
                                #get the actual action from the actual next_point
                                action_true= get_action_from_next_point(underlying_data.copy(),[next_point])
                                predicted_next_dict[k] = (action_true in action_predicted)
                            predicted = pd.concat([predicted,pd.DataFrame(predicted_next_dict, index=[0])], ignore_index=True)
                            sorted_prob = probability_of_next_point.sort_values(ascending=False)
                            rank, = np.where(sorted_prob.index.values == next_point)
                            rank_predicted.append(rank[0] + 1)

                    results['user'] = user_name
                    results['threshold'] = thres
                    if len(predicted) > 0:
                        results[f'ncp-{1}'] = predicted[1].sum()/len(predicted[1])
                    else:
                        results[f'ncp-{1}'] = None

                    results['rank'] = rank_predicted
                    hmm_results = pd.concat([hmm_results,pd.DataFrame(results)], ignore_index=True)


            hmm_results.to_csv("Experiments_Folder/VizRec/{}/{}/{}.csv".format(dataset, task, 'Weighted-KNN'), index=False)
            print("HMM results saved for task ", task, " and dataset ", dataset)