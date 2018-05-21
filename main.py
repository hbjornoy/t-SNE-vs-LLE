# EXECUTABLE WITH PLOTS FOR REPORT
"""
For computational reasons this file will highly depend on stored data from pickle which is a way to store python objects.

for each section there exist a jupyterfile with the same name, which might contain more analysis and interactive plots

INSTALLMENTS:
sklearn
matplotlib
seaborn
pickle

Running main.py:
1. install requirements

2. ensure all the data from the original folder is there

3. run main.py from same folder

4. You must exit a plot-window before pressing enter in terminal for it to work properly :)

"""


# IMPORTS
import matplotlib.pyplot as plt
from sklearn import manifold

# personal code imports
import helpers as HL
import plot_functions as PL
import pickle_functions as PK



#####################################################################################################
# Section III B1


#####################################################################################################
# Section III B2


#####################################################################################################
# Section III C


#####################################################################################################
# Section III D


#####################################################################################################
# Section III E


#####################################################################################################
# Section III F


print("#################################\nSECTION IV:")
print("Here we will analyse LLE and t-SNE on MNIST, a dataset with handwritten digits")
input("press enter to continue\n")

#constants
folder="mnist_pickles"
nb_samples = 1000
grid_width = 24

print("Importing data...")
inputs, targets = HL.import_mnist()
print("Printing samples from MNIST dataset...")
PL.plot_digits_samples(inputs, row_dim=5, col_dim=10)
input("press enter to continue\n")

print("################\nLLE: ")
print("Plotting heatmap analyzing different hyperparameters for LLE")
# change create=True if you want to compute the heatmap (it takes a while..)
test_lle_dict = PK.kmeans_clustering_f1_measure(inputs, targets, "lle", grid_width=grid_width, nb_samples=nb_samples,
                                         reg_range=(-8, 8), neighbor_range=(3,26), plot=True, create=False)
input("press enter to continue\n")

print("Plotting a couple of different LLE-embeddings")
# parameters of interest
nb_neighbors = (5, 19) 
regs =  (0.0182334800087, 2.46209240149e-07)
X_lle_5 = manifold.LocallyLinearEmbedding(nb_neighbors[0], 2, reg=regs[0], method='standard') \
                  .fit_transform(inputs[0:nb_samples])
X_lle_19 = manifold.LocallyLinearEmbedding(nb_neighbors[1], 2, reg=regs[1], method='standard') \
                  .fit_transform(inputs[0:nb_samples])
#parameters:
im = True # whether or not to show images
t = 9e-3 # how close the images should be shown

print("An example LLE embedding:")
fig = plt.figure(figsize=(7.5,7.5))
ax = PL.plot_embedding(inputs, X_lle_19, targets[0:nb_samples], fig=fig, subplot_pos=111, images=im, im_thres=t, title="Number of neighbors: "+str(nb_neighbors[1])+", Regularization: "+str(HL.round_sig(regs[1])))
plt.show()
input("press enter to continue\n")

print("The best LLE-embedding:")
fig = plt.figure(figsize=(7.5,7.5))
ax = PL.plot_embedding(inputs, X_lle_5, targets[0:nb_samples], fig=fig, subplot_pos=111, images=im, im_thres=t, title="Number of neighbors: "+str(nb_neighbors[0])+", Regularization: "+str(HL.round_sig(regs[0])))
plt.show()
input("press enter to continue\n")

print("################\nt-SNE: ")
print("Plotting heatmap analyzing different hyperparameters for t-SNE")
tsne_dict = PK.kmeans_clustering_f1_measure(inputs, targets, "tsne", grid_width=grid_width, nb_samples=nb_samples,
                                          min_grad_norm_range=(-8,0), perplexity_range=(2,100), plot=True, create=False)
input("press enter to continue\n")

print("Plotting a couple of different t-SNE-embeddings")
print("...this might take a minute...")
# parameters of interest
perplexity = (6, 82)
min_grad_norm = 4.961947603e-08

X_tsne_6 = manifold.TSNE(n_components=2, perplexity=perplexity[0], min_grad_norm=min_grad_norm, init='random', random_state=123)\
                   .fit_transform(inputs[0:nb_samples])
X_tsne_82 = manifold.TSNE(n_components=2, perplexity=perplexity[1], min_grad_norm=min_grad_norm, init='random', random_state=123)\
                   .fit_transform(inputs[0:nb_samples])
#parameters:
im = True # whether or not to show images
t = 1e-2 # how close the images should be shown

print("An example t-SNE embedding:")
fig = plt.figure(figsize=(7.5,7.5))
ax1 = PL.plot_embedding(inputs, X_tsne_82, targets[0:nb_samples], fig=fig, subplot_pos=111, images=im, im_thres=t, title="Perplexity: "+str(perplexity[1])+", Min_grad_norm: "+str(HL.round_sig(min_grad_norm)))
plt.show()
input("press enter to continue\n")

print("The best t-SNE-embedding:")
fig = plt.figure(figsize=(7.5,7.5))
ax1 = PL.plot_embedding(inputs, X_tsne_6, targets[0:nb_samples], fig=fig, subplot_pos=111, images=im, im_thres=t, title="Perplexity: "+str(perplexity[0])+", Min_grad_norm: "+str(HL.round_sig(min_grad_norm)))
plt.show()

print("\nThat's it, thank you for staying until the end.\nSAYONARA :D")
