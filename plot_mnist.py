
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import helpers as HL
SMALL_SIZE = 15
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_heatmap(acc_list, algorithm, param1_space, param2_space):
    """plot heatmap of accuracy with regard to different hyperparameters
  
    Parameters
    -------------
    
    Output
    -------------
    """
    fig, ax = plt.subplots(figsize=(10,8))
    ax = sns.heatmap(acc_list, cmap="YlGnBu_r", ax=ax, cbar_kws={'label': 'F1-score'})
    if algorithm == "lle":
        ax.set_xlabel("Regularization term (R)")
        ax.set_ylabel("Number of Neighbors (K)")
    elif algorithm == "tsne":
        ax.set_xlabel("Tolerance (tol)")
        ax.set_ylabel("Perplexity (Perp)")
    ax.set_xticklabels(HL.round_array(param2_space), rotation=90)
    ax.set_yticklabels(param1_space, rotation=0)
    plt.tight_layout()
    plt.savefig("images/MNIST_heatmap_" + algorithm)
    plt.show()

  