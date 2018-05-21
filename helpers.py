#  IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array, check_random_state, shuffle
from sklearn import datasets 
from math import log10, floor

import plot_functions as PL


# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D

np.random.seed(123)

def round_sig(x, sig=4):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def round_array(X): 
    new_x=np.zeros(X.shape)
    for i, x in enumerate(X): 
        new_x[i]=round_sig(x)
    return new_x


def import_mnist():
    """
    
     Output
    -------------
    """
    #import data
    digits = datasets.fetch_mldata('MNIST original', data_home="mnist")

    # shuffle data
    inputs, targets = shuffle(digits.data, digits.target, random_state=123)
    print("inputs shape: ", inputs.shape)
    print("targets shape: ", targets.shape)
    
    return inputs, targets


def make_swissroll(n=1000, noise=0.1, nb_holes=0, sigma=0.4, threshold=False, random_state=123,distribution='uniform'):
    """
    Make a swissroll data 
    
    Parameters
    ----------
    n : number of starting datapoints
    noise : sigma og gaussian noise in the data
    nb_holes : number of holes to make in the dataset
    sigma : sigma of the gaussian that makes the holes
    threshold : range [0-1], threshold decided the probabilitylimit if to keep or reject a point(nearby hole).
                Leaving it False makes use of a gaussian to basically downsample some holes randomly distributed.
    randomstate : any integer makes the code reprodusible.
    
    Returns
    -------
    X : The datapoints row=datapoints, col= dimensions
    t : color of the different points
    """
    generator = check_random_state(random_state)
    data_2d = make_2d_data(n, generator, distribution=distribution)
    
    #data_2d = scale_2d_data(data_2d)
    
    # add potential holes in data
    if nb_holes > 0:
        data_2d = make_2d_holes(np.squeeze(data_2d.T), nb_holes=nb_holes, sigma=sigma,threshold=threshold).T
        
    X, t = transform_to_3d(data_2d)
    
    # add potiential noise to data
    if noise > 0:
        X += noise * generator.randn(3, t.size)
    
    X = X.T
    t = np.squeeze(t)
    return X, t, data_2d


def make_2d_data(n, generator, distribution):
    """ generate a 2d dataset, sampled according to 'distibution'"""
    if distribution=='uniform': 
        vector1=generator.rand(1, n)
        vector2=generator.rand(1, n)
        
    elif distribution=='normal':
        vector1=generator.normal(0.5,scale=0.30,size=n)
        vector2=generator.normal(0.5,scale=0.30,size=n)

    elif distribution=='beta': 
       
        vector1=generator.beta(0.5,0.2,size=n)
        vector2=generator.beta(0.5,0.2,size=n)
       
    elif distribution=='mixed_normal': 
        
        a=generator.normal(0.25,0.125,size=int(n/2))
        b=generator.normal(0.75,0.125,size=int(n/2))
        vector1=np.vstack([a, b]).flatten().reshape(1,n)
        vector2=generator.rand(1, n)
       
    if distribution is not 'uniform': 
        scaler =MinMaxScaler(feature_range=(0,1))
        vector1=scaler.fit_transform(vector1.reshape(n,1)).flatten()
        vector2=scaler.fit_transform(vector2.reshape(n,1)).flatten()
    
    t = 1.5 * np.pi * (1 + 2 * vector1.reshape(1,n))
    y = 4.5 * np.pi * (1 + 2 * vector2.reshape(1,n))
    return np.array([t, y])


def transform_to_3d(data_2d):
    """ transform 2D data to 3D with a swiss roll transformation """
    t, y = data_2d[0].reshape(1,-1), data_2d[1].reshape(1,-1)
    x = t * np.cos(t)
    z = t * np.sin(t)
    return np.concatenate((x, y, z)), t


def plot_2d(data,color):
    """ plots 2D data as scatterplot"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[0], data[1],c=color, cmap=plt.cm.Spectral)
    plt.tight_layout()


def plot_3d(data, color):
    """ 
    plots 3D data as scatterplot where you can choose interactive plot or not
    The magic %matplotlib lines are not so stable so run this function in its own process
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
    plt.tight_layout()
    plt.show()
    

def keep_points(distance, sigma, threshold=False):
    """ Returns False if point is to be removed, True otherwise
    distance: a vector of distances to a given point. 
    sigma: variance of the gaussian
    threshold: if a value is given, then all distances that have a higher probability than threshold to be removed, 
    are removed. 
    """
    n=len(distance)
    probability_of_rejecting=norm.pdf(distance, scale=sigma)/norm.pdf(0,scale=sigma)
    if not threshold:
        return np.random.binomial(1,probability_of_rejecting)==0
    else: 
        return probability_of_rejecting < threshold
    
    
def make_2d_holes(data, nb_holes=3, sigma=0.1, threshold=False):
    """
    makes circular holes in 2d data if threshold is set it makes a holes with strict boundaries.
    With threshold to false it every points distance from the hole center is evaluated with a gaussian with mean=0.
    This makes it a randomly place downsampling/hole in the data.
    """
    #create distance matrix
    dist_mat = euclidean_distances(data,data)
    
    # choose the points where hole originate nb_holes
    indexes = np.arange(data[:,0].size)
    hole_centers = np.random.choice(indexes, nb_holes, replace=False)
    
    bool_list = list()
    # remove points nearby with probability
    for index in hole_centers:
        boolean_reject = keep_points(dist_mat[index], sigma, threshold=threshold)
        bool_list.append(boolean_reject)
    # put do OR operation on all the rejected point before removing
    bool_array = np.array(bool_list)
    remove_mask = np.all(bool_array, axis=0)
    
    data = data[remove_mask]
    return data


def get_X_with_label(inputs, targets, label):
    """ Get data with specific label in MNIST dataset"""
    y_bool = np.equal(targets,label)
    X_label = inputs[y_bool]
    return X_label


def get_differences(X_2d,trans):
    """
    Parameters
    -------------
    
    Output
    -------------
    """
    dist_mat_true = euclidean_distances(np.squeeze(X_2d).T,np.squeeze(X_2d).T)
    scaled_dist_mat_true=(dist_mat_true -np.min(dist_mat_true ))/(np.max(dist_mat_true )-np.min(dist_mat_true ))
    differences=np.zeros(len(trans))
    for i in range(len(trans)): 
        dist_mat_trans = euclidean_distances(trans[i],trans[i])
        scaled_dist_mat_trans=(dist_mat_trans -np.min(dist_mat_trans ))/(np.max(dist_mat_trans )-np.min(dist_mat_trans ))
        differences[i]=np.linalg.norm(scaled_dist_mat_true-scaled_dist_mat_trans)
    return differences

