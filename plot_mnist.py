
import seaborn as sns
import matplotlib.pyplot as plt




def plot_heatmap(acc_list, algorithm, param1_space, param2_space):
    """plot heatmap of accuracy with regard to different hyperparameters
  
    Parameters
    -------------
    
    Output
    -------------
    """
    fig, ax = plt.subplots(figsize=(15,15))
    ax = sns.heatmap(acc_list, cmap="YlGnBu_r", ax=ax, cbar_kws={'label': 'F1-score'})
    if algorithm == "lle":
        ax.set_xlabel("regularization term (R)")
        ax.set_ylabel("number of neighbors (K)")
    elif algorithm == "tsne":
        ax.set_xlabel("tolerance (tol)")
        ax.set_ylabel("perplexity (Perp)")
    ax.set_xticklabels(param2_space, rotation=90)
    ax.set_yticklabels(param1_space, rotation=0)
    plt.savefig("images/MNIST_heatmap_" + algorithm)
    plt.show()

  