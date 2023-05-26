import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import misc

class plotter():
    def __init__(self,user_threshold):
        self.y_accu_all = []
        self.thresholds=user_threshold


    def plot_user_stats(self,accuracies,user,noshow=False):
        plt.plot(self.thresholds, accuracies, label=user, marker='*')
        mean_y_accu = np.mean(accuracies)
        plt.axhline(mean_y_accu, color='black', linestyle='--', )


    def plot_main(self,accuracy,user):
        self.y_accu_all.append(accuracy)
        self.plot_user_stats(accuracy,user)



    def aggregate(self,accuracies,source):
        plt.yticks(np.arange(0.0, 1.0, 0.1))

        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        title = source
        mean_y_accu = np.mean([element for sublist in accuracies for element in sublist])
        plt.axhline(mean_y_accu, color='red', linestyle='-',label="Average: "+ "{:.2%}".format(mean_y_accu)  )
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        location = 'figures/Naive/' + title
        plt.savefig(location, bbox_inches='tight')
        plt.close()