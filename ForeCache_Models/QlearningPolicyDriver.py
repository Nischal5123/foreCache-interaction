
import sys
sys.path.append("..")
from QLearningPolicy import FiniteQLearningModel as QLearning
import forecache_environment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


env = forecache_environment.environment3()
users = env.user_list
accuracies=[]
plot_list=[]
vals, names, xs = [],[],[]
      # adds jitter to the data points - can be adjusted
for user in range(len(users)):
        names.append(user)
        plot_list.append(user)
        print("For user: ", users[user])
        threshold = 0.8
        env.process_data(users[user], threshold)

        # WARNING: If you try to set eps to a very low value,
        # And you attempt to get the m.score() of m.pi, there may not
        # be guarranteed convergence.
        eps = 5000
        S = [0,1,2]
        A = [0,1]
        START_EPS = 0.05
        m = QLearning(S, A, epsilon=START_EPS)

        SAVE_FIG = False
        history = []

        for i in range(1, eps+1):
            ep = []
            prev_observation = env.reset()
            prev_action = m.choose_action(m.b, prev_observation)

            total_reward = 0
            while True:
                # Run simulation
                next_observation, reward, done, _ = env.step(prev_action)
                next_action = m.choose_action(m.b, next_observation)
                m.update_Q((prev_observation, prev_action, reward, next_observation, next_action))

                prev_observation = next_observation
                prev_action = next_action

                total_reward += reward
                if done:
                    break

            history.append(total_reward)

        cumulative_test_reward, test_accuracy = m.score(env, m.b, n_samples=1000)  # how many times to sample or repeat to get mean of rewards
        vals.append(test_accuracy)
        xs.append(np.random.normal(user + 1, 0.04, len(test_accuracy)))
        print("Test accuracy:", np.mean(test_accuracy))
        env.reset(True, False)
        accuracies.append(np.mean(test_accuracy))
# plt.plot(plot_list, accuracies, '-ro', label='Q learning Average Test Accuracy for Users 1-20')
# plt.xlabel("Users 1 - 20")
# plt.ylabel("Test Accuracy on action prediction")
# plt.legend(loc='upper left')
# # plt.savefig("TDLearning-NoTestLearning-Decaying-Epsilon-" + str(epsi) + ".png")
# plt.clf()
##### Set style options here #####
sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
flierprops = dict(marker='o', markersize=1,
                  linestyle='none')
whiskerprops = dict(color='#00145A')
capprops = dict(color='#00145A')
medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')
plt.boxplot(vals, labels=names)
for x, val in zip(xs, vals):
    plt.scatter(x, val, alpha=0.4)
plt.xlabel("Users", fontweight='normal', fontsize=14)
plt.ylabel("Accuracy", fontweight='normal', fontsize=14)
sns.despine(bottom=True) # removes right and top axis lines
plt.axhline(y=np.mean(accuracies), color='#ff3300', linestyle='--', linewidth=1, label='Global Average')
plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)
plt.show()
