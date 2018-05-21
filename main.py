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


"""


# IMPORTS


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
input("\npress enter to continue\n")

#constants
folder="mnist_pickles"
nb_samples = 1000
grid_width = 24

print("importing data...")
inputs, targets = HL.import_mnist()
print("printing samples from MNIST dataset...")
PL.plot_digits_samples(inputs, row_dim=5, col_dim=10)
input("\npress enter to continue\n")

print("##############\nLLE: ")
print("plotting heatmap analyzing different hyperparameters for LLE")
# change create=True if you want to compute the heatmap (it takes a while..)
test_lle_dict = PK.kmeans_clustering_f1_measure(inputs, targets, "lle", grid_width=grid_width, nb_samples=nb_samples,
                                         reg_range=(-8, 8), neighbor_range=(3,26), plot=True, create=False)
input("\npress enter to continue\n")

print("plotting four different LLE-embeddings, whereas K=5 and R=0.01823 is the best embedding")
nb_components = 2
# parameters of interest
nb_neighbors = (5,19)
regs = (2.46209240149e-07, 0.0182334800087)

X_lle_1 = manifold.LocallyLinearEmbedding(nb_neighbors[0], nb_components, reg=regs[0], method='standard') \
                  .fit_transform(inputs[0:nb_samples])
X_lle_2 = manifold.LocallyLinearEmbedding(nb_neighbors[0], nb_components, reg=regs[1], method='standard') \
                  .fit_transform(inputs[0:nb_samples])
X_lle_3 = manifold.LocallyLinearEmbedding(nb_neighbors[1], nb_components, reg=regs[0], method='standard') \
                  .fit_transform(inputs[0:nb_samples])
X_lle_4 = manifold.LocallyLinearEmbedding(nb_neighbors[1], nb_components, reg=regs[1], method='standard') \
                  .fit_transform(inputs[0:nb_samples])





