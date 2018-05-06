import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from ipywidgets import *
import pickle
from matplotlib import offsetbox
import seaborn as sns


def plot_inter(color,var,Z,i,variable,transformation,error=None,times=None,difference=None,error_type=None):
    """
    color: the colors of Z
    var: a list of the variable that is changing
    Z: A list of t-SNE transformations, with different perplexities s.t. Z[i] has perplexity per[i]. 
    i: the index of the transformation we want to plot
    variable: 'per', 'threshold', 'learning_rate'
    """
    if variable=='per':
        print('The perplexity is', var[i])
    if variable=='learning_rate':
        print('The learning rate is', var[i])
    if variable=='threshold':
        print('The threshold is', var[i])
    if variable=='early_exaggeration': 
        print('The early exaggeration is', var[i])
    if variable=='n_neighbors': 
        print('The n_neighbors is', var[i])   
    if variable=='reg': 
        print('The regularization term is', var[i]) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(transformation)
    ax.scatter(Z[i][:, 0], Z[i][:, 1], c=color, cmap=plt.cm.Spectral)
    if error is not None:
        plot_error_dist_and_time(var, error,times,difference,variable,error_type=error_type, i=i)
    plt.show()

def plot_error_dist_and_time(var, error,times,difference,variable='variable', filename=False, error_type=False, i=False):
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(131)
    ax.plot(var,error,'go--')
    if i: 
        ax.axvline(x=var[i],  color='r', linestyle='--')
    if (variable=="threshold" or variable=='reg'):
        plt.xscale('log')
    #ax.set_title("t-SNE, KL divergence")
    if error_type:
        ax.set_ylabel(error_type)
    else:
        ax.set_ylabel('Error')
    ax.set_xlabel('%s' %variable)
    ax = fig.add_subplot(132)
    ax.plot(var,times,'go--')
    if i: 
        ax.axvline(x=var[i],  color='r', linestyle='--')
    if (variable=="threshold" or variable=='reg'):
        plt.xscale('log')
    ax.set_ylabel('Time')
    ax.set_xlabel('%s' %variable)
    ax = fig.add_subplot(133)
    ax.plot(var,difference,'go--')
    if i: 
        ax.axvline(x=var[i],  color='r', linestyle='--')
    if (variable=="threshold" or variable=='reg'):
        plt.xscale('log')
    ax.set_ylabel('Difference in 2d distance')
    ax.set_xlabel('%s' %variable)
    #ax.set_title("t-SNE, Computational time")
    plt.show()
    if filename: 
        plt.savefig(filename)
def plot_and_save_tsne(perplexity, filename, Z, per, color):
    if np.argwhere(per==perplexity).flatten(): 
        i=np.argwhere(per==perplexity).flatten()[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("t-SNE, perplexity=%i" %perplexity)
        ax.scatter(Z[i][:, 0], Z[i][:, 1], c=color, cmap=plt.cm.Spectral)
        plt.savefig(filename)
        plt.show()
    else: 
        print('Transformation is not made for this perplexity, availiable perplexities are:', per)
        
# Scale and visualize the embedding vectors
def plot_embedding(X_orig, X_trans, y, title=None):
    """
    Plots the manifold embedding with the some of the original images across the data.
    Strongly inspired and based on sklearn docs examples:
    http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    # Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
    #          Olivier Grisel <olivier.grisel@ensta.org>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #          Gael Varoquaux
    # License: BSD 3 clause (C) INRIA 2011
    """
    x_min, x_max = np.min(X_trans, 0), np.max(X_trans, 0)
    X_trans = (X_trans - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X_trans.shape[0]):
        plt.text(X_trans[i, 0], X_trans[i, 1], str(int(y[i])),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    
    # the pictures are too big
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1, 1]])  # just something big
        for i in range(X_trans.shape[0]):
            dist = np.sum((X_trans[i] - shown_images) ** 2, 1)
            if np.min(dist) < 3e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X_trans[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_orig[i].reshape((int(np.sqrt(X_orig.shape[1])),int(np.sqrt(X_orig.shape[1])))), 
                                      cmap=plt.cm.gray_r, zoom=0.6),
            X_trans[i])
            ax.add_artist(imagebox)
    
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
def plot_heatmap(acc_list, algorithm, param1_space, param2_space):
    """plot heatmap of accuracy with regard to different hyperparameters"""
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