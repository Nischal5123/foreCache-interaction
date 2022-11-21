import environment2Location as environment2
import numpy as np
from collections import defaultdict
import pdb
import misc
import matplotlib.pyplot as plt


class NaiveProbabilistic:
    def __init__(self):
        self.freq = defaultdict(lambda: defaultdict(float))
        self.reward = defaultdict(lambda: defaultdict(float))

    def NaiveProbabilistic(self, user, env, thres):

        # for t in itertools.count():
        # print(u)
        length = len(env.mem_action)
        # pdb.set_trace()
        threshold = int(length * thres)

        # for i in range(1, length-1):
        #     print("{} {}".format(env.mem_states[i-1], env.mem_action[i]))

        for i in range(1, threshold):
            self.freq[env.mem_states[i]][env.mem_action[i-1]] += 1
            self.reward[env.mem_states[i - 1]][env.mem_action[i]] += env.mem_reward[i]

        # Normalizing to get the probability
        for states in self.freq:
            sum = 1
            for actions in self.freq[states]:
                sum += self.freq[states][actions]
            for actions in self.freq[states]:
                self.freq[states][actions] = self.freq[states][actions] / sum
                # self.freq[states][actions] /= sum

        # Debugging probablity calculation
        # for states in self.freq:
        #     for actions in self.freq[states]:
        #         print("{} {} {}".format(states, actions, self.freq[states][actions]))

        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        for i in range(threshold + 1, length - 1):
            try:
                _max = max(self.freq[env.mem_states[i - 1]], key=self.freq[env.mem_states[i - 1]].get)
                if _max == env.mem_action[i] and self.freq[env.mem_states[i - 1]][_max] > 0:
                    # print(env.mem_states[i-1], _max, self.freq[env.mem_states[i-1]][_max], env.mem_action[i], self.freq[env.mem_states[i-1]])
                    accuracy += 1
            except ValueError:
                pass
            denom += 1
        accuracy /= denom
        # print("Accuracy {} {:.2f}".format(user, accuracy))
        obj = misc.misc([])
        print("{}, {:.2f}".format(obj.get_user_name(user), accuracy))
        self.freq.clear()
        self.reward.clear()
        return accuracy


if __name__ == "__main__":

    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    user_list_3D = env.user_list_3D

    user_list_experienced = np.array(
        ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_82316e37-1117-4663-84b4-ddb6455c83b2.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_ff56863b-0710-4a58-ad22-4bf2889c9bc0.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_bda49380-37ad-41c5-a109-7fa198a7691a.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_3abeecbe-327a-441e-be2a-0dd3763c1d45.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_954edb7c-4eae-47ab-9338-5c5c7eccac2d.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_a6aab5f5-fdb6-41df-9fc6-221d70f8c6e8.csv',
         'data/NDSI-2D\\taskname_ndsi-2d-task_userid_8b544d24-3274-4bb0-9719-fd2bccc87b02.csv'])
    # only training users
    # user_list_2D = ['data/NDSI-2D\\taskname_ndsi-2d-task_userid_44968286-f204-4ad6-a9b5-d95b38e97866.csv',
    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_3abeecbe-327a-441e-be2a-0dd3763c1d45.csv',
    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_6d49fab8-273b-4a91-948b-ecd14556b049.csv',
    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_72a8d170-77ae-400e-b2a5-de9e1d33a714.csv',
    #                 'data/NDSI-2D\\taskname_ndsi-2d-task_userid_733a1ac5-0b01-485e-9b29-ac33932aa240.csv']
    user_list_first_time = np.setdiff1d(user_list_2D, user_list_experienced)
    total = 0
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    obj2 = misc.misc([])
    # for u in user_list_3D[:10]:
    #     y_accu = []
    #     for thres in threshold:
    #         env.process_data(u, 0)
    #         obj = NaiveProbabilistic()
    #         accu = obj.NaiveProbabilistic(u, env, thres)
    #         total += accu
    #         y_accu.append(accu)
    #         env.reset(True, False)
    #     print("User ", obj2.get_user_name(u), " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))
    #
    #     plt.plot(threshold, y_accu, label=obj2.get_user_name(u))
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # title = "NDSI-3D-3-STATES-1"
    # # pdb.set_trace()
    # plt.title(title)
    # location = 'figures/Naive/' + title
    # plt.savefig(location, bbox_inches='tight')
    # plt.close()
    #
    # for u in user_list_3D[10:]:
    #     y_accu = []
    #     for thres in threshold:
    #         env.process_data(u, 0)
    #         obj = NaiveProbabilistic()
    #         accu = obj.NaiveProbabilistic(u, env, thres)
    #         total += accu
    #         y_accu.append(accu)
    #         env.reset(True, False)
    #     print("User ", obj2.get_user_name(u), " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))
    #
    #     plt.plot(threshold, y_accu, label=obj2.get_user_name(u))
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # title = "NDSI-3D-3-STATES-2"
    # # pdb.set_trace()
    # plt.title(title)
    # location = 'figures/Naive/' + title
    # plt.savefig(location, bbox_inches='tight')
    # plt.close()
    # # print(total / (len(users_b) + len(users_f)))
    for u in user_list_first_time[:4]:
        y_accu = []
        for thres in threshold:
            env.process_data(u, 0)
            obj = NaiveProbabilistic()
            accu = obj.NaiveProbabilistic(u, env, thres)
            total += accu
            y_accu.append(accu)
            env.reset(True, False)
        print("User ",obj2.get_user_name(u)," across all thresholds " , "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=obj2.get_user_name(u))
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    title = "NDSI-2D-3-STATES-1"
    # pdb.set_trace()
    plt.title(title)
    location = 'figures/Naive/' + title
    plt.savefig(location, bbox_inches='tight')
    plt.close()

    for u in user_list_experienced[:4]:
        y_accu = []
        for thres in threshold:
            env.process_data(u, 0)
            obj = NaiveProbabilistic()
            accu = obj.NaiveProbabilistic(u, env, thres)
            total += accu
            y_accu.append(accu)
            env.reset(True, False)
        print("User ", obj2.get_user_name(u), " across all thresholds ", "Global Accuracy: ", np.mean(y_accu))

        plt.plot(threshold, y_accu, label=obj2.get_user_name(u))
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    title = "NDSI-2D-3-STATES-2"
    # pdb.set_trace()
    plt.title(title)
    location = 'figures/Naive/' + title
    plt.savefig(location, bbox_inches='tight')
    plt.close()

