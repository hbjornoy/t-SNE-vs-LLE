
import seaborn as sns




def plot_heatmap(acc_list, algorithm, param1_space, param2_space):
    """plot heatmap of accuracy with regard to different hyperparameters
  
    Parameters
    -------------
    
    Output
    -------------
    """

    ax = sns.heatmap(acc_list, cmap="YlGnBu_r")
    if algorithm == "lle":
        ax.set_xlabel("regularization term (R)")
        ax.set_ylabel("number of neighbors (K)")
    elif algorithm == "tsne":
        ax.set_xlabel("tolerance (tol)")
        ax.set_ylabel("perplexity (Perp)")
    ax.set_xticklabels(param2_space)
    ax.set_yticklabels(param1_space)
    plt.show()

  