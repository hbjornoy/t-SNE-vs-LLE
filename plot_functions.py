import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D
from ipywidgets import *
import pickle
from matplotlib import offsetbox


np.random.seed(123)


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


def plot_inter_grid(colors,var1,var2, Z,j, i,data_augmentation,variable,transformation,error=None,times=None,difference=None,error_type=None):
    """
    colors: A list of the colors of each dataset 
    var1: a list of the first variable that is changing (eg noise, holes...))
    var2: a list of the second variable that is changing (eg reg, number of neighbours, perplexity)
    Z: A list of a list of transformed data. Z[var1[0]][var2[0]] gives the transformation with for example noise 0 and regularisation 0
    j: the index of the first variable that we want to plot
    i: the index of the second variable that we want to plot
    data_augmentation: Description of the difference in dataset, eg 'noise', 'holes'.. 
    variable: Name of the hyperparameter we are changing (var2),'per', 'threshold', 'learning_rate'
    transformation: 'lle' or 't-sne'
    error: 
    """
    str_holes=['1: 1 hole, size 2', '2: 1 hole, size 5','3: 2 holes, size 2', '4: 2 holes, size 5','5: 3 holes, size 2', '6: 3 holes, size 5'] 
    if data_augmentation=='noise':
        print('The noise is ', var1[j])
    elif data_augmentation=='distribution':
        distributions=['uniform','normal','mixed_normal','beta']
        print('The distribution is ', distributions[j])
    elif data_augmentation=='datapoints':
        print('The number of datapoints is ', var1[j])
    elif data_augmentation=='holes':
        #print('The type of hole(s) is', var1[j])
        print(str_holes[j])
    if variable=='per':
        print('The perpelxity is', var2[i])
    elif variable=='learning_rate':
        print('The learning rate is', var2[i])
    elif variable=='threshold':
        print('The threshold is', var2[i])
    elif variable=='early_exaggeration': 
        print('The early exaggeration is', var2[i])
    elif variable=='n_neighbors': 
        print('The n_neighbors is', var2[i])   
    elif variable=='reg': 
        print('The regularization term is', var2[i]) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(transformation)
    ax.scatter(Z[j][i][:, 0], Z[j][i][:, 1], c=colors[j], cmap=plt.cm.Spectral)
    if error is not None:
        plot_error_dist_and_time(var1, error[:,i],times[:,i],difference[:,i],data_augmentation,error_type=error_type, i=j)
    plt.show()


def plot_error_dist_and_time(var, error,times,difference,variable='variable', filename=False, error_type=False, i=False):
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(131)
    ax.plot(var,error,'go--')
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
    ax.axvline(x=var[i],  color='r', linestyle='--')
    if (variable=="threshold" or variable=='reg'):
        plt.xscale('log')
    ax.set_ylabel('Time, s')
    ax.set_xlabel('%s' %variable)
    ax = fig.add_subplot(133)
    ax.plot(var,difference,'go--')
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
    """ usikker på om jeg faktisk bruker denne..  """
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
        

     
def plot_augmented_swissrolls(Xs, colors, var, variable_name):
    fig = plt.figure(figsize=(15,10))
    
    for i in range(len(Xs)):
        X=Xs[i]
        ax = fig.add_subplot(230+i+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[i], cmap=plt.cm.Spectral)
        if variable_name is 'distribution':
            ax.set_title( var[i] +'-distributed points')
        else: 
            ax.set_title(variable_name+': %1.2f' %var[i])
    plt.show()

    
def plot_digits_samples(inputs, row_dim, col_dim):
    
    # calculate pixelwidth
    pixel_width = int(np.sqrt(inputs.shape[1]))
    image = np.zeros(((pixel_width+2)*row_dim, (pixel_width+2)*col_dim))
    
    # map the inputs "unstructured" data to the two-dimensional picture
    for i in range(row_dim):
        x = (pixel_width+2)*i + 1

        for j in range (col_dim):
            y = (pixel_width+2)*j + 1
            image[x:x+pixel_width, y:y+pixel_width] = inputs[i*col_dim + j].reshape((pixel_width,pixel_width))

    plt.imshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title("Samples from MNIST, handwritten digits")
    plt.savefig("mnist_examples_of_data.pdf")
    plt.show()

