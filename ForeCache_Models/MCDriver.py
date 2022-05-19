import forecache_environment
# The typical imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from MonteCarlo import FiniteMCModel as MC

env = forecache_environment.environment3()
users = env.user_list
accuracies=[]
plot_list=[]


for i in range(len(users)):
    print("For user: ", users[i])
    threshold=0.8
    env.process_data(users[i],threshold)
    plot_list.append(users[i][28:-10])




    eps = 10000 #how many episodes to train for
    # S = [(x, y, z) for x in range(4,22) for y in range(1,11) for z in [True,False]]
    # A = 2

    S = [0,1,2]
    A = [0,1]

    m = MC(S, A, epsilon=1)
    for i in range(1, eps+1):
        ep = []
        observation = env.reset()
        while True:
            # Choosing behavior policy
            action = m.choose_action(m.b, observation)
            # Run simulation
            next_observation, reward, done, _ = env.step(action)
            ep.append((observation, action, reward))
            observation = next_observation
            if done:
                break

        m.update_Q(ep)
        # Decaying epsilon, reach optimal policy
        m.epsilon = max((eps-i)/eps, 0.1)
    cumulative_test_reward , test_accuracy =m.score(env, m.pi, n_samples=100)# how many times to sample or repeat to get mean of rewards
    env.reset(True, False)
    print("Final expected returns : {}".format(cumulative_test_reward))

    #plt.plot(range(len(test_accuracy)),test_accuracy)
    #plt.show()
    mean_acc=np.mean(test_accuracy)
    accuracies.append(mean_acc)
    print("Test Accuracy : {}".format(mean_acc * 100))
plt.figure(figsize=[10,10])
plt.plot(range(len(plot_list)), accuracies, '-ro', label='Monte Carlo Test Accuracy for Users 0-19')
plt.xticks(range(len(plot_list)), rotation='vertical')
plt.margins(0.002)
plt.xlabel("Users 1 - 20")
plt.ylabel("Test Accuracy on action prediction")
plt.legend(loc='upper right')
plt.show()