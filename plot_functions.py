import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from ipywidgets import *
import pickle
def plot_inter(color,var,Z,i,variable,transformation):
    """
    color: the colors of Z
    var: a list of the variable that is changing
    Z: A list of t-SNE transformations, with different perplexities s.t. Z[i] has perplexity per[i]. 
    i: the index of the transformation we want to plot
    variable: 'per', 'threshold', 'learning_rate'
    """
    if variable=='per':
        print('The perpelxity is', var[i])
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
    plt.show()

def plot_error_and_time(var, error,times,variable='variable', filename=False, error_type=False):
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121)
    ax.plot(var,error,'go--')
    if (variable=="Threshold" or variable=='reg'):
        plt.xscale('log')
    ax.set_title("t-SNE, KL divergence")
    if error_type:
        ax.set_ylabel(error_type)
    else:
        ax.set_ylabel('Error')
    ax.set_xlabel('%s' %variable)
    ax = fig.add_subplot(122)
    ax.plot(var,times,'go--')
    if (variable=="Threshold" or variable=='reg'):
        plt.xscale('log')
    ax.set_ylabel('Time')
    ax.set_xlabel('%s' %variable)
    ax.set_title("t-SNE, Computational time")
    
    if filename: 
        plt.savefig(filename)
def plot_and_save_tsne(perplexity, filename, Z=pickle.load(open("p_Z_tsne.pkl", "rb")), per=pickle.load(open("per.pkl", "rb")), color=pickle.load(open("color.pkl", "rb"))):
    if np.argwhere(per==perplexity).flatten(): 
        i=np.argwhere(per==perplexity).flatten()[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("t-SNE, perplexity=%i" %perplexity)
        ax.scatter(Z[i][:, 0], Z[i][:, 1], c=color, cmap=plt.cm.Spectral)
        plt.savefig(filename)
        plt.show()
    else: 
        print('Transformation is not made for this perpexity, availiable perplexities are:', per)