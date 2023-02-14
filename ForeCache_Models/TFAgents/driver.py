import forecache_d3rl as environment2
import numpy as np
from collections import Counter
import d3rlpy
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discrete_action_match_scorer

from d3rlpy.ope import DiscreteFQE #The off-policy evaluation is a method to estimate the trained policy performance only with offline datasets.
import plotting

np.random.seed(0)
DISCRETE_ALGORITHMS= {
    "bc": 'DiscreteBC',
    "bcq": 'DiscreteBCQ',
    "cql": 'DiscreteCQL',
    "dqn": 'DQN',
    "double_dqn": 'DoubleDQN',
    "nfq": 'NFQ',
    "sac": 'DiscreteSAC',
    "random": 'DiscreteRandomPolicy'
}


def set_user():
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    user_list_experienced = np.array(
        ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
    user_list_first_time = np.setdiff1d(user_list_2D, user_list_experienced)
    return user_list_first_time, user_list_experienced


def getMdpDataset(user,episode_size=5):
    env=environment2.environment2()
    env.process_data(user,0)

    threshold=get_threshold(env.mem_roi)
    # steps of observations with shape of (100,)
    observations = np.array((env.mem_states))
    # steps of actions with shape of (4,)
    actions = np.array(env.mem_action)
    # steps of rewards
    rewards = np.array(env.mem_reward)
    # steps of terminal flags
    terminals = np.array(np.zeros(len(env.mem_states)-1))
    terminals=np.append(terminals, 1)

    #divide episode without terminal states
    episode_terminals=[]
    for step in range(len(env.mem_states)):
        if step%episode_size ==0:
            episode_terminals.append(1)
        else:
            episode_terminals.append(0)



    dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals,  episode_terminals=np.array(episode_terminals))

    return dataset,threshold


def algorithm(algo,dataset,user,train_threshold=0.5,epochs=50):
    # split train and test episodes
    train_episodes, test_episodes = train_test_split(dataset, test_size=1-train_threshold)

    agent = d3rlpy.algos.create_algo(algo,True)


    # start training
    metrics=agent.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=epochs,
            scorers={
                # 'advantage': discounted_sum_of_advantage_scorer, # smaller is better
                # 'td_error': td_error_scorer, # smaller is better
                # #:https://d3rlpy.readthedocs.io/en/latest/references/generated/d3rlpy.metrics.scorer.td_error_scorer.html
                # 'value_scale': average_value_estimation_scorer, # smaller is better,
                'action-match': discrete_action_match_scorer
            })

    # off-policy evaluation algorithm
    fqe = DiscreteFQE(algo=agent)
    # train estimators to evaluate the trained policy
    evaluation_metrics= fqe.fit(test_episodes,
            eval_episodes=test_episodes,
            n_epochs=1,
            scorers={
                # This metrics suggests how different the greedy-policy is from the given episodes in discrete action-space.
                # If the given episdoes are near-optimal, the large percentage would bebetter.
                'action-match': discrete_action_match_scorer
            },
              experiment_name= str( algo + "_" + user + "_" + str(train_threshold)))


    return evaluation_metrics[len(evaluation_metrics)-1][1]['action-match']





def get_user_name(url):
    string = url.split('\\')
    fname = string[len(string) - 1]
    uname = ((fname.split('userid_')[1]).split('.'))[0]
    return uname

def get_threshold(roi):
    counts = Counter(roi)
    proportions = []
    total_count = len(roi)

    for i in range(1, max(counts.keys()) + 1):
        current_count = sum(counts[key] for key in range(1, i + 1))
        proportions.append(current_count / total_count)
    return proportions[:-1]






def main(algo):
    aggregate_plotter = plotting.plotter(None)
    user_list_non_exp,user_list_exp=set_user()
    y_accu_all=[]
    for user in user_list_exp[:6]:
        user_name=get_user_name(user)
        dataset,threshold= getMdpDataset(user)
        plotter=plotting.plotter(threshold)
        y_accu=[]
        for thres in threshold:
            accuracy=algorithm(algo,dataset,user_name,train_threshold=thres,epochs=50)
            y_accu.append(accuracy)
            print(
                "# User :{}, Threshold : {}, Accuracy: {}".format(user_name, thres, accuracy))
        plotter.plot_main(y_accu, user_name[:5])
        y_accu_all.append(y_accu)
    title = "experienced"+ str(DISCRETE_ALGORITHMS[algo])
    aggregate_plotter.aggregate(y_accu_all, title)



if __name__ == '__main__':

        main(algo='cql')









# def off policy_eval(trained_algorithm):
#     # off-policy evaluation algorithm
#     fqe = DiscreteFQE(algo=trained_algorithm)
#
#     # train estimators to evaluate the trained policy
#     fqe.fit(train_episodes,
#             eval_episodes=test_episodes,
#             n_epochs=10,
#             scorers={
#                'init_value': initial_state_value_estimation_scorer,
#                'soft_opc': soft_opc_scorer(return_threshold=600)
#             })


