import numpy as np
import pickle
import time
import helpers as HL
from sklearn.manifold import TSNE as t_sne
from sklearn.manifold import LocallyLinearEmbedding as lle
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics

import plot_functions as PL
from plot_mnist import plot_heatmap

np.random.seed(123)


def get_swiss_roll(folder,modification='', create=False, n=1000, noise=0.01):
    """
    Function used to create and load the swiss roll used in section B
    
    Parameters
    -------------
    folder: the name of the folder the pickles are/should be saved in
    modification: a string, if you want to modify the name of the pickles
    create: True if you want to create datasets, false if you want to load pickels.
    n: Default 1000
    noise: Default 0.01
    
    Output
    -------------
    color: The corresponding colors
    X: The 3d swiss roll
    X_2d: The corresponding 2d data
    """
    if create: 
        X, color, X_2d=HL.make_swissroll(n=n, noise=noise)
        pickle.dump( X, open(folder+"/X"+modification+".pkl", "wb")) 
        pickle.dump( color, open(folder+"/color"+modification+".pkl", "wb")) 
        pickle.dump( X_2d, open(folder+"/X_2d"+modification+".pkl", "wb")) 
    else: 
        color=pickle.load(open(folder+"/color"+modification+".pkl", "rb"))
        X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        X_2d=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    return color,X,X_2d


def get_augmented_swissroll(create=False,noise=False,holes=False,datapoints=False,density=False,n=1000):
    """
    Function used to creat and load the swiss rolls used in section C, D E and F. 
    Parameters
    -------------
    create: True if you want to create the datasets, false if you want to load pickles. Default False
    noise: True if you want datasets with different levels of noise. Defualt : False
    holes: True if you want datasets with different holes.  Default :False
    datapoints: True if you want datasets with different number of datapoints.  Default: False
    density: True if you want datasets with different underlying densities.  Default False
    n: Number of datapoints.  Default 1000.
    
    Output
    -------------
    
    colors: A list of the corresponding colors
    Xs: A list of the 3d swiss rolls
    X_2ds: A list of the corresponding 2d datasets
  
    noises: a vector containing the different noise levels
    or 
    holes: a vector containing tuples (a,b) descibing the holes, where a is the number of holes and b is the size of the holes
    or
    datapoint:a vector containing the number of datapoints
    or
    distributions: a vector containing strigs that are the name of the underlying distributions. 
    """

    folder='Data'
    if noise: 
        noises=[0.05,0.1,0.5,1,2]
        name='noise'
      
        N=len(noises)
    elif holes: 
        name='holes'
        holes=[[1,2],[1,5],[2,2],[2,5],[3,2],[3,5]]
        N=len(holes)
    elif datapoints: 
        name='datapoints'
        datapoint=[100,500,1000,5000,10000]
        N=len(datapoint)
    elif density: 
        name='density'
        distributions=['uniform','normal','mixed_normal','beta']
        N=len(distributions)
        
    if create: 
        colors=[]
        Xs=[]
        X_2ds=[]
        for i in range(N):
        
            if noise: 
                X, color, X_2d=X, color, X_2d=HL.make_swissroll(n=n, noise=noises[i], random_state=123)
            elif holes: 
                X, color, X_2d=X, color, X_2d=HL.make_swissroll(n=n, noise=0.1, nb_holes=holes[i][0], sigma=holes[i][1],
                                                            threshold=0.5, random_state=123)
            elif datapoints: 
                X, color, X_2d=X, color, X_2d=HL.make_swissroll(n=datapoint[i],random_state=123)
            elif density: 
                X, color, X_2d=X, color, X_2d=HL.make_swissroll(n=n,random_state=123, distribution=distributions[i])
             
            colors.append(color)
            Xs.append(X)
            X_2ds.append(X_2d)
        pickle.dump( Xs, open(folder+"/Xs_"+name+".pkl", "wb")) 
        pickle.dump( colors, open(folder+"/colors_"+name+".pkl", "wb")) 
        pickle.dump( X_2ds, open(folder+"/X_2ds_"+name+".pkl", "wb")) 
      
    else: 
        colors=pickle.load(open(folder+"/colors_"+name+".pkl", "rb"))
        Xs=pickle.load(open(folder+"/Xs_"+name+".pkl", "rb"))
        X_2ds=pickle.load(open(folder+"/X_2ds_"+name+".pkl", "rb"))
    if noise: 
        return Xs, colors, X_2ds,noises
    elif holes: 
        return Xs, colors, X_2ds,holes
    elif datapoints: 
        return Xs, colors, X_2ds,datapoint
    elif density: 
        return Xs, colors, X_2ds,distributions

    
def perplexity(folder=None,modification='',per=np.arange(2,150,2), create=False, pkl=True, X=None, X_2d_tsne=None): 
    """
    Function used to generate or load t-SNE transformations with a range of different perplexities. 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    per: the perplexities you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl: true if you want to make a pickle for every value of per, otherwise false
    X: the data you want to transform
    X_2d_tsne: the original 2D version of X
    
    Output
    -------------
    p_Z: A ilst of the t-SNE transformations
    per: a vector with the corresponding values of perplexity
    p_times: a vector with the corresponding values of computational time
    p_kl_divergence: a vector with the corresponding values kl divergence
    p_differences: a vector with the corresponding values of difference in 2d distance
    
    """
    if create: 
        p_Z=[]
        p_times=np.zeros(len(per))
        p_kl_divergence=np.zeros(len(per))
        if X is None: 
            X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        for i, p in enumerate(per):
            tsne=t_sne(perplexity=p, random_state=123)
            start_time=time.time()
            p_Z.append(tsne.fit_transform(X))
            p_times[i]= time.time()-start_time
            p_kl_divergence[i]=tsne.kl_divergence_
        if pkl: 
            pickle.dump( p_Z, open(folder+"/p_Z_tsne"+modification+".pkl", "wb")) 
            pickle.dump(per, open(folder+"/per"+modification+".pkl","wb"))
            pickle.dump(p_times, open(folder+"/p_times"+modification+".pkl","wb"))
            pickle.dump(p_kl_divergence, open(folder+"/p_kl_divergence"+modification+".pkl","wb"))
    else: 
        p_Z= pickle.load(open(folder+"/p_Z_tsne"+modification+".pkl", "rb"))
        per=pickle.load(open(folder+"/per"+modification+".pkl", "rb"))
        p_times=pickle.load(open(folder+"/p_times"+modification+".pkl", "rb"))
        p_kl_divergence=pickle.load(open(folder+"/p_kl_divergence"+modification+".pkl", "rb"))
    if X_2d_tsne is None:
        X_2d_tsne=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    p_differences=HL.get_differences(X_2d_tsne,p_Z)
    return p_Z,per,p_times,p_kl_divergence,p_differences


def early_exaggeration(folder,modification='',create=False,early_exaggeration=np.arange(1,80,1)): 
    """
    Function used to generate or load t-SNE transformations with a range of different early exaggeration rates. 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    early_exaggeration: the early exaggeration rates you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl true if you want to make a pickle for every value of early_exaggeration, otherwise false 
    
    Output
    -------------
    e_Z: A ilst of the t-SNE transformations
    early_exaggeration: a vector with the corresponding values of early_exaggeration
    e_times: a vector with the corresponding values of computational time
    e_kl_divergence: a vector with the corresponding values kl divergence
    e_differences: a vector with the corresponding values of difference in 2d distance
    """
    if create: 
        e_Z=[]
        e_times=np.zeros(len(early_exaggeration))
        e_kl_divergence=np.zeros(len(early_exaggeration))
        X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        for i, e in enumerate(early_exaggeration):
            tsne=t_sne(early_exaggeration=e,random_state=123)
            start_time=time.time()
            e_Z.append(tsne.fit_transform(X))
            e_times[i]= time.time()-start_time
            e_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( e_Z, open(folder+"/e_Z_tsne"+modification+".pkl", "wb")) 
        pickle.dump(early_exaggeration, open(folder+"/early_exaggeration"+modification+".pkl","wb"))
        pickle.dump(e_times, open(folder+"/e_times"+modification+".pkl","wb"))
        pickle.dump(e_kl_divergence, open(folder+"/e_kl_divergence"+modification+".pkl","wb"))
    else: 
        e_Z= pickle.load(open(folder+"/e_Z_tsne"+modification+".pkl", "rb"))
        early_exaggeration=pickle.load(open(folder+"/early_exaggeration"+modification+".pkl", "rb"))
        e_times=pickle.load(open(folder+"/e_times"+modification+".pkl", "rb"))
        e_kl_divergence=pickle.load(open(folder+"/e_kl_divergence"+modification+".pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    e_differences=HL.get_differences(X_2d_tsne,e_Z)
    return e_Z,early_exaggeration,e_times,e_kl_divergence,e_differences


def learning_rates(folder,modification='', create=False, learning_rates=np.arange(5,1000,5) ): 
    """
    Function used to generate or load t-SNE transformations with a range of different learning rates. 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    learning_rates: the learning rates rates you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl: true if you want to make a pickle for every value of learning_rate, otherwise false 
    
    Output
    -------------
    l_Z: A ilst of the t-SNE transformations
    learning_rates: a vector with the corresponding values of learning rate
    l_times: a vector with the corresponding values of computational time
    l_kl_divergence: a vector with the corresponding values kl divergence
    l_differences: a vector with the corresponding values of difference in 2d distance
    """
    if create: 
        l_Z=[]
        l_times=np.zeros(len(learning_rates))
        l_kl_divergence=np.zeros(len(learning_rates))
        X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        for i, l in enumerate(learning_rates):
            tsne=t_sne(learning_rate=l,random_state=123)
            start_time=time.time()
            l_Z.append(tsne.fit_transform(X))
            l_times[i]= time.time()-start_time
            l_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( l_Z, open(folder+"/l_Z_tsne"+modification+".pkl", "wb")) 
        pickle.dump(learning_rates, open(folder+"/learning_rates"+modification+".pkl","wb"))
        pickle.dump(l_times, open(folder+"/l_times"+modification+".pkl","wb"))
        pickle.dump(l_kl_divergence, open(folder+"/l_kl_divergence"+modification+".pkl","wb"))
    else: 
        l_Z= pickle.load(open(folder+"/l_Z_tsne"+modification+".pkl", "rb"))
        learning_rates=pickle.load(open(folder+"/learning_rates"+modification+".pkl", "rb"))
        l_times=pickle.load(open(folder+"/l_times"+modification+".pkl", "rb"))
        l_kl_divergence=pickle.load(open(folder+"/l_kl_divergence"+modification+".pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    l_differences=HL.get_differences(X_2d_tsne,l_Z)
    return l_Z,learning_rates,l_times,l_kl_divergence,l_differences


def threshold(folder, modification='',create=False,threshold=np.logspace(-14,-1,50) ): 
    """
    Function used to generate or load t-SNE transformations with a range of different thresholds (tol/min_grad_norm). 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    threshold: the thresholds you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl: true if you want to make a pickle for every value of threshold, otherwise false 
    
    Output
    -------------
    t_Z: A ilst of the t-SNE transformations
    learning_rates: a vector with the corresponding values of learning rate
    t_times: a vector with the corresponding values of computational time
    t_kl_divergence: a vector with the corresponding values kl divergence
    t_differences: a vector with the corresponding values of difference in 2d distance
    """
    if create: 
        t_Z=[]
        t_times=np.zeros(len(threshold))
        t_kl_divergence=np.zeros(len(threshold))
        X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        for i, t in enumerate(threshold):
            tsne=t_sne(min_grad_norm=t, random_state=123)
            start_time=time.time()
            t_Z.append(tsne.fit_transform(X))
            t_times[i]= time.time()-start_time
            t_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( t_Z, open(folder+"/t_Z_tsne"+modification+".pkl", "wb")) 
        pickle.dump(threshold, open(folder+"/threshold"+modification+".pkl","wb"))
        pickle.dump(t_times, open(folder+"/t_times"+modification+".pkl","wb"))
        pickle.dump(t_kl_divergence, open(folder+"/t_kl_divergence"+modification+".pkl","wb"))
    else: 
        t_Z= pickle.load(open(folder+"/t_Z_tsne"+modification+".pkl", "rb"))
        threshold=pickle.load(open(folder+"/threshold"+modification+".pkl", "rb"))
        t_times=pickle.load(open(folder+"/t_times"+modification+".pkl", "rb"))
        t_kl_divergence=pickle.load(open(folder+"/t_kl_divergence"+modification+".pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    t_differences=HL.get_differences(X_2d_tsne,t_Z)
    return t_Z,threshold,t_times,t_kl_divergence,t_differences


def n_neighbors(folder=None, modification='',create=False,n_neighbors=np.arange(3,60,1), X=None, pkl=True, X_2d_lle=None): 
    """
    Function used to generate or load LLE transformations with a range of different number of neighbours. 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    n_neighbors: the number of neighbours you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl: true if you want to make a pickle for every value of early_exaggeration, otherwise false 
    X: Dataset (3d swiss roll)
    X_2d_lle: corresponding 2d data
    
    Output
    -------------
   
    n_Y: A ilst of the LLE transformations
    n_neighbors: a vector with the corresponding values of number of neighbours
    n_times:a vector with the corresponding values of computational time
    n_reconstruction_error: a vector with the corresponding values of reconstruction error
    n_differences:a vector with the corresponding values of difference in 2d distance
    """
    if create: 
        n_components=2
        if X is None:
            X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        n_Y=[]
        n_times=np.zeros(len(n_neighbors))
        n_reconstruction_error=np.zeros(len(n_neighbors))
        for i, n in enumerate(n_neighbors):
            LLE=lle(n, n_components,eigen_solver='auto')
            start_time=time.time()
            n_Y.append(LLE.fit_transform(X))
            n_times[i]= time.time()-start_time
            n_reconstruction_error[i]=LLE.reconstruction_error_ 
        if pkl: 
            pickle.dump( n_Y, open(folder+"/n_Y_lle"+modification+".pkl", "wb")) 
            pickle.dump(n_neighbors, open(folder+"/n_neighbors"+modification+".pkl","wb"))
            pickle.dump(n_times, open(folder+"/n_times"+modification+".pkl","wb"))
            pickle.dump(n_reconstruction_error, open(folder+"/n_reconstruction_error"+modification+".pkl","wb"))
    else: 
        n_Y= pickle.load(open(folder+"/n_Y_lle"+modification+".pkl", "rb"))
        #lle_color=pickle.load(open("lle_color.pkl", "rb"))
        n_neighbors=pickle.load(open(folder+"/n_neighbors"+modification+".pkl", "rb"))
        n_times=pickle.load(open(folder+"/n_times"+modification+".pkl", "rb"))
        n_reconstruction_error=pickle.load(open(folder+"/n_reconstruction_error"+modification+".pkl", "rb"))
    if X_2d_lle is None:
        X_2d_lle=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    n_differences=HL.get_differences(X_2d_lle,n_Y)
    return n_Y,n_neighbors, n_times,n_reconstruction_error,n_differences


def n_reg(folder=None, modification='',create=False,reg=np.logspace(-14,10,50), X=None, pkl=True, X_2d_lle=None): 
    """
    Function used to generate or load LLE transformations with a range of different regularization term. 
    Parameters
    -------------
    folder: The name of the folder the pickles should be put in / are in
    modification: the modification done to the dataset, a string used in the name of the corresponding pickles. 
    reg: the values of the regularizatin term you want to create transformations of
    create: true is you want to create the transformations, false if you want to load them
    pkl: true if you want to make a pickle for every value of early_exaggeration, otherwise false 
    X: Dataset (3d swiss roll)
    X_2d_lle: corresponding 2d data
    
    Output
    -------------
   
    r_Y: A ilst of the LLE transformations
    reg: a vector with the corresponding values of the regularisation term
    r_times:a vector with the corresponding values of computational time
    r_reconstruction_error: a vector with the corresponding values of reconstruction error
    r_differences:a vector with the corresponding values of difference in 2d distance
    """
    if create:
        if X is None:
            X=pickle.load(open(folder+"/X"+modification+".pkl", "rb"))
        n_components=2
        neighbors=12
        r_Y=[]
        r_times=np.zeros(len(reg))
        r_reconstruction_error=np.zeros(len(reg))
        for i, r in enumerate(reg):
            LLE=lle(neighbors, n_components,reg=r,eigen_solver='auto')
            start_time=time.time()
            r_Y.append(LLE.fit_transform(X))
            r_times[i]= time.time()-start_time
            r_reconstruction_error[i]=LLE.reconstruction_error_ 
        if pkl: 
            pickle.dump( r_Y, open(folder+"/r_Y_lle"+modification+".pkl", "wb")) 
            pickle.dump(reg, open(folder+"/reg"+modification+".pkl","wb"))
            pickle.dump(r_times, open(folder+"/r_times"+modification+".pkl","wb"))
            pickle.dump(r_reconstruction_error, open(folder+"/r_reconstruction_error"+modification+".pkl","wb"))
    else: 
        r_Y= pickle.load(open(folder+"/r_Y_lle"+modification+".pkl", "rb"))
        reg=pickle.load(open(folder+"/reg"+modification+".pkl", "rb"))
        r_times=pickle.load(open(folder+"/r_times"+modification+".pkl", "rb"))
        r_reconstruction_error=pickle.load(open(folder+"/r_reconstruction_error"+modification+".pkl", "rb"))
    if X_2d_lle is None:
        X_2d_lle=pickle.load(open(folder+"/X_2d"+modification+".pkl", "rb"))
    r_differences=HL.get_differences(X_2d_lle,r_Y)
    return r_Y,reg, r_times,r_reconstruction_error,r_differences

from sklearn.metrics import f1_score

def kmeans_clustering_f1_measure(inputs, targets, algorithm, grid_width=4, nb_samples=1000, reg_range=(-12, 12), 
                               neighbor_range=(3,35), min_grad_norm_range=(-4, -1), perplexity_range=(3,100), plot=False,
                               create=False, folder="mnist_pickles", name="normal", random_state=123):
    """
    This function performs manifold fitting and transformation of the input data. Performs kmeans and using only information about clusterlabels from kmeans classifies with Support Vector Classification(SVC). This is done for different parameters in the for of a grid-search. Afterwards the data is stored and plotted as a heatmap if requested. If created before one can load pickle.
    
    Parameters
    -------------
    inputs: Manifold-embedded data in 2D
    targets: labels of inputs
    algorithm: string with name of algorithm, valid input: "lle" and "tsne"
    grid_width: the number of elements we want to gridsearch from each of the to parameters
    nb_samples: original MNIST data has 70.000 samples, only use subset to be able to compute
    reg_range: relevant for LLE, the range of regularization term parameter with logistic range
    neighbor_range: relevant for LLE, the range of number of neighbors parameter with linear range
    min_grad_norm_range: relevant for TSNE, the range of min_grad_norm parameter with logistic range
    perplexity_range: relevant for TSNE, the range of perplexity parameter with linear range
    plot: whether to plot the heatmap or not
    create: if true it calculates the kmeans accuracy for every parameter combination and stores it. If false it tries to load one with the same parameters.
    folder: the folder name to store the pickles
    name: custom name to append to picklename
    random_state: seed for stochastic algorihtms
    
    Output
    -------------
    return_dict: a dictionary with
        - "acc_list": contains 2D ndarray with accuracy scores for different parameters. row = param1 , col=param2.
        - "algorithm": the algorihtm of your choosing from parameters
        - "param1_space": the values of linear parameter for either lle or tsne
        - "param2_space": the values of logistic parameter for either lle or tsne
    
    """
    if not create:
        # load pickle
        d = pickle.load(open(folder+"/F1_"+algorithm+"_grid-"+str(grid_width)+"_samples-"+str(nb_samples)+"_"+name+".pkl", "rb"))
        # plot heatmap of accuracy with regard to different hyperparameters
        plot_heatmap(d['f1_list'], d['algorithm'], d['param1_space'], d['param2_space'])
        # return data from pickled data
        return d
    
    nb_clusters = 10
    nb_components = 2
    y_true = targets[0:nb_samples]
    
    if algorithm=="lle":
        # LLE params
        param1_space = np.linspace(neighbor_range[0], neighbor_range[1], num=grid_width, endpoint=True, dtype=int)
        param2_space = np.logspace(reg_range[0], reg_range[1], grid_width, endpoint=True)
    elif algorithm=="tsne":
        # tsne params
        param1_space = np.linspace(perplexity_range[0], perplexity_range[1], num=grid_width, endpoint=True, dtype=int)
        param2_space = np.logspace(min_grad_norm_range[0],min_grad_norm_range[1],grid_width, endpoint=True)
    else:
        print("Manifold valid input is either lle or tsne")
        raise
        
    f1_list = np.zeros((grid_width,grid_width))
    
    for i, param1 in enumerate(param1_space):
        for j, param2 in enumerate(param2_space):
            
            if algorithm=="lle":
                mani = manifold.LocallyLinearEmbedding(param1, nb_components, reg=param2,
                                                       method='standard', random_state=random_state)
            elif algorithm=="tsne":
                mani = manifold.TSNE(n_components=2, perplexity = param1, min_grad_norm=param2, 
                                     n_iter=1000, metric='euclidean', init='random', verbose=0, random_state=random_state)
                
            X = mani.fit_transform(inputs[0:nb_samples])
            
            # calculate F1-score for 10 different kmeans with different seeds from 0 to 10 then average scores
            # this is to deminish the effect of kmeans initialization
            kmeans_f1s = 0
            number_of_seeds = 10
            for seed in range(0,number_of_seeds):

                # Kmeans
                kmeans = KMeans(init='random', n_clusters=nb_clusters, random_state=seed).fit(X)
                # Make the kmeans labels the only data to classify upon
                X = kmeans.labels_[:,np.newaxis]

                cluster_sizes = list()
                f1_score = 0
                # count cluster_sizes
                for cluster in range(0,10):
                    cluster_sizes.append(sum(cluster==clabel for clabel in kmeans.labels_))

                # calculate F1-score for clustering by maximizing f1-score for each individual class
                for digit in range(0,10):
                    digit_size = sum(digit==label for label in y_true)
                    f1_scores = np.zeros(10)
                    for cluster in range(0,10):
                        nb_correct = sum(y_true[kmeans.labels_==cluster] == digit)
                        # Check if nb_correct is zero to not divide on zero
                        if (nb_correct == 0):
                            f1_scores[cluster] = 0
                        else:
                            recall = nb_correct / digit_size
                            precision = nb_correct / cluster_sizes[cluster]
                            f1_scores[cluster] = ( (2*recall*precision) / (recall + precision))
                    f1_score += (digit_size/nb_samples)*np.max(f1_scores)
                kmeans_f1s += f1_score
            f1_list[i,j] = kmeans_f1s/number_of_seeds          
            
    if plot:
        # plot heatmap of accuracy with regard to different hyperparameters
        plot_heatmap(f1_list, algorithm, param1_space, param2_space)
    
    return_dict = {"f1_list": f1_list, "algorithm": algorithm, "param1_space": param1_space, "param2_space": param2_space}
    # pickle it
    pickle.dump(return_dict, open(folder+"/F1_"+algorithm+"_grid-"+str(grid_width)+"_samples-"+str(nb_samples)+"_"+name+".pkl", "wb")) 
    
    return return_dict

##########
########## MUST REMOVE before delivery  #######   #######   #######   #######
##########
def kmeans_clustering_accuracy(inputs, targets, algorithm, grid_width=4, nb_samples=1000, reg_range=(-12, 12), 
                               neighbor_range=(3,35), min_grad_norm_range=(-4, -1), perplexity_range=(3,100), plot=False,
                               create=False, folder="mnist_pickles", name="normal", random_state=123):
    """
    This function performs manifold fitting and transformation of the input data. Performs kmeans and using only information about clusterlabels from kmeans classifies with Support Vector Classification(SVC). This is done for different parameters in the for of a grid-search. Afterwards the data is stored and plotted as a heatmap if requested. If created before one can load pickle.
    
    Parameters
    -------------
    inputs: Manifold-embedded data in 2D
    targets: labels of inputs
    algorithm: string with name of algorithm, valid input: "lle" and "tsne"
    grid_width: the number of elements we want to gridsearch from each of the to parameters
    nb_samples: original MNIST data has 70.000 samples, only use subset to be able to compute
    reg_range: relevant for LLE, the range of regularization term parameter with logistic range
    neighbor_range: relevant for LLE, the range of number of neighbors parameter with linear range
    min_grad_norm_range: relevant for TSNE, the range of min_grad_norm parameter with logistic range
    perplexity_range: relevant for TSNE, the range of perplexity parameter with linear range
    plot: whether to plot the heatmap or not
    create: if true it calculates the kmeans accuracy for every parameter combination and stores it. If false it tries to load one with the same parameters.
    folder: the folder name to store the pickles
    name: custom name to append to picklename
    random_state: seed for stochastic algorihtms
    
    Output
    -------------
    return_dict: a dictionary with
        - "acc_list": contains 2D ndarray with accuracy scores for different parameters. row = param1 , col=param2.
        - "algorithm": the algorihtm of your choosing from parameters
        - "param1_space": the values of linear parameter for either lle or tsne
        - "param2_space": the values of logistic parameter for either lle or tsne
    
    """
    if not create:
        # load pickle
        d = pickle.load(open(folder+"/"+algorithm+"_grid-"+str(grid_width)+"_samples-"+str(nb_samples)+"_"+name+".pkl", "rb"))
        # plot heatmap of accuracy with regard to different hyperparameters
        plot_heatmap(d['acc_list'], d['algorithm'], d['param1_space'], d['param2_space'])
        # return data from pickled data
        return d
    
    nb_clusters = 10
    nb_components = 2
    
    if algorithm=="lle":
        # LLE params
        param1_space = np.linspace(neighbor_range[0], neighbor_range[1], num=grid_width, endpoint=True, dtype=int)
        param2_space = np.logspace(reg_range[0], reg_range[1], grid_width, endpoint=True)
    elif algorithm=="tsne":
        # tsne params
        param1_space = np.linspace(perplexity_range[0], perplexity_range[1], num=grid_width, endpoint=True, dtype=int)
        param2_space = np.logspace(min_grad_norm_range[0],min_grad_norm_range[1],grid_width, endpoint=True)
    else:
        print("Manifold valid input is either lle or tsne")
        raise
        
    acc_list = np.zeros((grid_width,grid_width))
    
    for i, param1 in enumerate(param1_space):
        for j, param2 in enumerate(param2_space):
            
            if algorithm=="lle":
                mani = manifold.LocallyLinearEmbedding(param1, nb_components, reg=param2,
                                                       method='standard', random_state=random_state)
            elif algorithm=="tsne":
                mani = manifold.TSNE(n_components=2, perplexity = param1, min_grad_norm=param2, 
                                     n_iter=1000, metric='euclidean', init='random', verbose=0, random_state=random_state)
                
            X = mani.fit_transform(inputs[0:nb_samples])
                
            # Kmeans
            kmeans = KMeans(init='k-means++', n_clusters=nb_clusters, random_state=123).fit(X)
            # Make the kmeans labels the only data to classify upon
            X = kmeans.labels_[:,np.newaxis]
            
            # Match up the clusterlabel and with the semantic meaning of handwritten number
            # classify with SVC, rbf kernel. convex and nice. 
            model = SVC()
            model.fit(X, targets[0:nb_samples])
            y_pred = model.predict(X)
            
            # calculate accuracy (only from kmeans clustering info)
            acc_list[i, j] = metrics.accuracy_score(targets[0:nb_samples], y_pred)
    if plot:
        # plot heatmap of accuracy with regard to different hyperparameters
        plot_heatmap(acc_list, algorithm, param1_space, param2_space)
    
    return_dict = {"acc_list": acc_list, "algorithm": algorithm, "param1_space": param1_space, "param2_space": param2_space}
    # pickle it
    pickle.dump(return_dict, open(folder+"/"+algorithm+"_grid-"+str(grid_width)+"_samples-"+str(nb_samples)+"_"+name+".pkl", "wb")) 
    
    return return_dict


def lle_different_data(var,folder, modification,N=None, Xs=None, X_2ds=None, create=False):
    """
    Parameters
    -------------
    var: 'r' or 'n' for regularization or number of neighbours
    folder: the folder the pickles should be in. 
    modification: 'noise', 'holes' etc
    N: number of datasets, (eg len(noises))
    Xs: list of 3d swiss rolls
    X_2ds: list of corresponding 2d data
    create: True if we want to create pickles
    
    Output
    -------------
    Ys: a list of the LLE transformations
    variable: a vector containing the values of 'var' 
    times: a vector with the corresponding values of computational time
    reconstruction_errors:  a vector with the corresponding values of reconstruction error
    difference:a vector with the corresponding values of difference in 2d distance

    """
    if var=='r':
        param=np.logspace(-14,10,50)
    elif var=='n':
        param=np.arange(3,60,1)
    if create: 
        Ys=[]
        times=np.zeros([N,len(param)])
        reconstruction_errors=np.zeros([N,len(param)])
        difference=np.zeros([N,len(param)])
        for i in range(len(Xs)): 
            if var=='r':
                Y,variable, time,reconstruction_error,differences=n_reg(create=True,
                                                                        X=Xs[i], reg=param,
                                                                        pkl=False,X_2d_lle=X_2ds[i])
            elif var=='n':
                 Y,variable, time,reconstruction_error,differences=n_neighbors(create=True, X=Xs[i],
                                                                               n_neighbors=param,pkl=False,
                                                                               X_2d_lle=X_2ds[i])
            Ys.append(Y)
            times[i,:]=time
            reconstruction_errors[i,:]=reconstruction_error
            difference[i,:]=differences
        pickle.dump(Ys, open(folder+"/"+var+"_Y_"+modification+".pkl", "wb")) 
        pickle.dump(times, open(folder+"/"+var+"_times_"+modification+".pkl","wb"))
        pickle.dump(reconstruction_errors, open(folder+"/"+var+"_reconstruction_errors_"+modification+".pkl","wb")) 
        pickle.dump(difference, open(folder+"/"+var+"_difference_"+modification+".pkl","wb"))
        
        if var=='r':
            pickle.dump(variable, open(folder+"/reg_"+modification+".pkl","wb"))
        elif var=='n':
            pickle.dump(variable, open(folder+"/neighbours_"+modification+".pkl","wb"))
        
    else: 
        Ys= pickle.load(open(folder+"/"+var+"_Y_"+modification+".pkl", "rb"))
        times=pickle.load(open(folder+"/"+var+"_times_"+modification+".pkl", "rb"))
        reconstruction_errors=pickle.load(open(folder+"/"+var+"_reconstruction_errors_"+modification+".pkl", "rb"))
        difference=pickle.load(open(folder+"/"+var+"_difference_"+modification+".pkl", "rb"))
        if var=='r':
            variable=pickle.load(open(folder+"/reg_"+modification+".pkl", "rb"))
        elif var=='n':
            variable=pickle.load(open(folder+"/neighbours_"+modification+".pkl", "rb"))
    return Ys,variable,times,reconstruction_errors,difference



def t_sne_different_data(var,folder, modification,N=None, Xs=None, X_2ds=None, create=False):
    """
    Parameters
    -------------
    var: 'p' for perplexity
    folder: the folder the pickles should be in. 
    modification: 'noise', 'holes' etc
    N: number of datasets, (eg len(noises))
    Xs: list of 3d swiss rolls
    X_2ds: list of corresponding 2d data
    create: True if we want to create pickles
    
    Output
    -------------
    Zs: a list of the t-SNE transformations
    param: a vector containing the values of 'var' 
    times: a vector with the corresponding values of computational time
    kl_divergence:  a vector with the corresponding values of kl divergence
    difference:a vector with the corresponding values of difference in 2d distance
    """
    if var=='p':
        param=np.arange(2,150,2)
    if create: 
        Zs=[]
        times=np.zeros([N,len(param)])
        kl_divergences=np.zeros([N,len(param)])
        difference=np.zeros([N,len(param)])
        for i in range(len(Xs)): 
            if var=='p':
                 Z,param,time,kl_divergence,differences=perplexity(X=Xs[i],per=param,
                                                               pkl=False, X_2d_tsne=X_2ds[i],create=True)
  
            Zs.append(Z)
            times[i,:]=time
            kl_divergences[i,:]=kl_divergence
            difference[i,:]=differences
        
        pickle.dump(Zs, open(folder+"/"+var+"_Z_"+modification+".pkl", "wb")) 
        pickle.dump(times, open(folder+"/"+var+"_times_"+modification+".pkl","wb"))
        pickle.dump(kl_divergences, open(folder+"/"+var+"_kl_divergences_"+modification+".pkl","wb")) 
        pickle.dump(difference, open(folder+"/"+var+"_difference_"+modification+".pkl","wb"))
        if var=='p':
            pickle.dump(param, open(folder+"/perplexity_"+modification+".pkl","wb"))

    else: 
        Zs= pickle.load(open(folder+"/"+var+"_Z_"+modification+".pkl", "rb"))
        times=pickle.load(open(folder+"/"+var+"_times_"+modification+".pkl", "rb"))
        kl_divergences=pickle.load(open(folder+"/"+var+"_kl_divergences_"+modification+".pkl", "rb"))
        difference=pickle.load(open(folder+"/"+var+"_difference_"+modification+".pkl", "rb"))
        if var=='p':
            param=pickle.load(open(folder+"/perplexity_"+modification+".pkl", "rb"))
          
    
    return Zs,param,times,kl_divergences,difference

