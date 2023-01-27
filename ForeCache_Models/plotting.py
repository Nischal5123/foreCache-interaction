import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_episode_stats(accuracy,num_episodes,source,noshow=False):
    # Plot the episode length over time
    x=range(num_episodes)
    y=accuracy
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(x,y)
    plt.axhline(np.mean(accuracy))
    plt.xlabel("Episode")
    plt.ylabel("Accuracies")
    plt.title("Episode Acuracies")

    filename = "figs/" + source + ".png"
    plt.savefig(filename)

    if noshow:
        plt.close(fig1)
    else:
        # plt.show(fig1)
        fig1.show()
