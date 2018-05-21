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

print("#################################\nComparison of LLE and t-SNE.\nHåvard Bjørnøy and Hedda Vik \nClose figures to proceed:")

#####################################################################################################
# Section III B

print("#################################\nSECTION III B:")
print("In this section we will analyse LLE and t-SNE on the original Swiss Roll.\nFirst we plot the dataset, in 3d and in 2d")
input("press enter to continue\n")

#constants: 
folder="SectionB"

#Importing data: 
color,X,X_2d=PK.get_swiss_roll(folder, create=False, n=1000, noise=0.01)

#Plotting Dataset: 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Data Set")
fig.patch.set_facecolor(color='w')
plt.show()

#Plotting the 2d dataset: 
HL.plot_2d(X_2d,color)
plt.show()
#####################################################################################################
# Section III B1

print("#################################\nSECTION III B1:")
print("Here we will analyse LLE on the original Swiss Roll")
input("press enter to continue\n")

print('Loading pickles with LLE transformations, for a range of K and R')
n_Y,n_neighbors, n_times,n_reconstruction_error,n_differences=PK.n_neighbors(folder=folder,create=False)
r_Y,reg, r_times,r_reconstruction_error,r_differences=PK.n_reg(folder,create=False)
print('Interactive plots are availiable in the corresponding jupyter notebook')
print('Plotting the most interesting transformations, as presented in the report')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(321)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, K=3')
ax.scatter(n_Y[0][:, 0], n_Y[0][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(323)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, K=13')
ax.scatter(n_Y[10][:, 0], n_Y[10][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(325)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, K=32')
ax.scatter(n_Y[29][:, 0], n_Y[29][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(322)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, R=2.80000 e-12')
ax.scatter(r_Y[5][:, 0], r_Y[5][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(324)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, R=0.000596')
ax.scatter(r_Y[22][:, 0], r_Y[22][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(326)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, R=1.59986')
ax.scatter(r_Y[29][:, 0], r_Y[29][:, 1], c=color, cmap=plt.cm.Spectral)
plt.tight_layout()
plt.show()

#####################################################################################################
# Section III B2

print("#################################\nSECTION III B2:")
print("Here we will analyse t-SNE on the original Swiss Roll")
input("press enter to continue\n")

print('Loading pickles with t-SNE transformations, for a range of Perp, E, L and tol')
p_Z,per,p_times,p_kl_divergence,p_differences=PK.perplexity(folder,create=False)
e_Z,early_exaggeration,e_times,e_kl_divergence,e_differences=PK.early_exaggeration(folder,create=False)
l_Z,learning_rates,l_times,l_kl_divergence,l_differences=PK.learning_rates(folder,create=False)
t_Z,threshold,t_times,t_kl_divergence,t_differences=PK.threshold(folder,create=False)
print('Interactive plots are availiable in the corresponding jupyter notebook')
print('Plotting the most interesting transformations, as presented in the report')

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(221)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Perp=2')
ax.scatter(p_Z[0][:, 0], p_Z[0][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(222)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Perp=16')
ax.scatter(p_Z[7][:, 0], p_Z[7][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(223)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Perp=110')
ax.scatter(p_Z[54][:, 0], p_Z[54][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(224)
ax.set_title('t-SNE, computational time')
ax.plot(per,p_times,'go--')  
ax.set_ylabel('Time, s')
ax.set_xlabel('Perp')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(222)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, tol=6.5e-05')
ax.scatter(t_Z[37][:, 0], t_Z[37][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(223)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, tol=7.5e-4')
ax.scatter(t_Z[41][:, 0], t_Z[41][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(221)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, tol=1e-14')
ax.scatter(t_Z[0][:, 0], t_Z[0][:, 1], c=color, cmap=plt.cm.Spectral)
ax = fig.add_subplot(224)
ax.set_title('t-SNE, computational time')
plt.xscale('log')
ax.plot(threshold,t_times,'go--')  
ax.set_ylabel('Time, s')
ax.set_xlabel('tol')
plt.tight_layout()
plt.show()
#####################################################################################################
# Section III C
print("#################################\nSECTION III C:")
print("Here we will analyse LLE and t-SNE on the Swiss Roll with noise")
input("press enter to continue\n")

#Defining constants: 
folder="SectionC_grid"
modification="noises"

print('Loading pickles with all datasets')
Xs, colors, X_2ds,noises=PK.get_augmented_swissroll(create=False, noise=True)
print('Plotting all the datasets')
PL.plot_augmented_swissrolls(Xs, colors, noises, 'Noise')

print('Loading pickles with LLE and t-SNE transformations, with different values of K, R and Perp')
n_Ys,neighbours,n_times,n_reconstruction_errors,n_difference=PK.lle_different_data('n',folder, modification,N=5,create=False,Xs=Xs, X_2ds=X_2ds)
r_Ys,reg,r_times,r_reconstruction_errors,r_difference=PK.lle_different_data('r',folder, modification,N=5,create=False,Xs=Xs, X_2ds=X_2ds)
p_Zs,per,p_times,p_kl_divergences,p_difference=PK.t_sne_different_data('p',folder, modification,N=5,create=False,Xs=Xs, X_2ds=X_2ds)

print('Interactive plots are availiable in the jupyter notebook section III C')
print('Plotting a comparison of the two algoritms, with optimal values for each of the hyperparametrs.\nAll other hyperparameters are set to their default value')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(531)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.05, K=8')
ax.scatter(n_Ys[0][5][:, 0], n_Ys[0][5][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(534)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.10, K=8')
ax.scatter(n_Ys[1][5][:, 0], n_Ys[1][5][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(537)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.50, K=9')
ax.scatter(n_Ys[2][6][:, 0], n_Ys[2][6][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,10)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=1.00, K=6')
ax.scatter(n_Ys[3][3][:, 0], n_Ys[3][3][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,13)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=2.00, K=7')
ax.scatter(n_Ys[4][4][:, 0], n_Ys[4][4][:, 1], c=colors[4], cmap=plt.cm.Spectral)
ax = fig.add_subplot(532)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.05, R=0.000193069772888')
ax.scatter(r_Ys[0][21][:, 0], r_Ys[0][21][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(535)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.10, R=0.00568986602902')
ax.scatter(r_Ys[1][24][:, 0], r_Ys[1][24][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(538)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=0.50, R=0.0175751062485')
ax.scatter(r_Ys[2][25][:, 0], r_Ys[2][25][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,11)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=1.00, R=0.00568986602902')
ax.scatter(r_Ys[3][24][:, 0], r_Ys[3][24][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,14)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Noise=2.00, R=0.00568986602902')
ax.scatter(r_Ys[4][24][:, 0], r_Ys[4][5][:, 1], c=colors[4], cmap=plt.cm.Spectral)
ax = fig.add_subplot(533)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Noise=0.05, Perp=16')
ax.scatter(p_Zs[0][7][:, 0], p_Zs[0][7][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(536)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Noise=0.10, Perp=18')
ax.scatter(p_Zs[1][8][:, 0], p_Zs[1][8][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(539)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Noise=0.50, Perp=22')
ax.scatter(p_Zs[2][10][:, 0], p_Zs[2][10][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,12)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Noise=1.00, Perp=24')
ax.scatter(p_Zs[3][11][:, 0], p_Zs[3][11][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(5,3,15)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Noise=2.00, Perp=32')
ax.scatter(p_Zs[4][15][:, 0], p_Zs[4][15][:, 1], c=colors[4], cmap=plt.cm.Spectral)
plt.tight_layout()
plt.show()

#####################################################################################################
# Section III D
print("#################################\nSECTION III D:")
print("Here we will analyse LLE and t-SNE on the Swiss Roll with holes")
input("press enter to continue\n")

#Defining constants: 
folder="SectionD_grid"
modification="holes"
str_holes=['1: 1 hole, size 2', '2: 1 hole, size 5','3: 2 holes, size 2', '4: 2 holes, size 5','5: 3 holes, size 2', '6: 3 holes, size 5']

print('Loading pickles with all datasets')
Xs, colors, X_2ds,holes=PK.get_augmented_swissroll(create=False, holes=True)

print('Plotting all the datasets')
PL.plot_augmented_swissrolls(Xs, colors, [1,2,3,4,5,6], 'Holes')
for i in range(len(str_holes)):
    print(str_holes[i])
    
for i, X_2d in enumerate(X_2ds): 
    print(str_holes[i])
    HL.plot_2d(X_2d,colors[i])
    plt.show()

print('Loading pickles with LLE and t-SNE transformations, with different values of K, R and Perp')
n_Ys,neighbours,n_times,n_reconstruction_errors,n_difference=PK.lle_different_data('n',folder, modification,N=6,create=False,Xs=Xs, X_2ds=X_2ds)
r_Ys,reg,r_times,r_reconstruction_errors,r_difference=PK.lle_different_data('r',folder, modification,N=6,create=False,Xs=Xs, X_2ds=X_2ds)
n_Ys,neighbours,n_times,n_reconstruction_errors,n_difference=PK.lle_different_data('n',folder, modification,N=6,create=False,Xs=Xs, X_2ds=X_2ds)
p_Zs,per,p_times,p_kl_divergences,p_difference=PK.t_sne_different_data('p',folder, modification,N=6,create=False,Xs=Xs, X_2ds=X_2ds)


print('Interactive plots are availiable in the jupyter notebook section III C')
print('Plotting a comparison of the two algoritms, with optimal values for each of the hyperparametrs.\nAll other hyperparameters are set to their default value')


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(631)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 1, K=14')
ax.scatter(n_Ys[0][11][:, 0], n_Ys[0][11][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(634)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 2, K=7')
ax.scatter(n_Ys[1][4][:, 0], n_Ys[1][4][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(637)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 3, K=10')
ax.scatter(n_Ys[2][7][:, 0], n_Ys[2][7][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,10)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 4, K=9')
ax.scatter(n_Ys[3][6][:, 0], n_Ys[3][6][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,13)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 5, K=12')
ax.scatter(n_Ys[4][9][:, 0], n_Ys[4][9][:, 1], c=colors[4], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,16)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 6, K=7')
ax.scatter(n_Ys[5][4][:, 0], n_Ys[5][4][:, 1], c=colors[5], cmap=plt.cm.Spectral)
ax = fig.add_subplot(632)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 1, R=0.00184206996933')
ax.scatter(r_Ys[0][23][:, 0], r_Ys[0][23][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(635)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 2, R=0.00184206996933')
ax.scatter(r_Ys[1][23][:, 0], r_Ys[1][23][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(638)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 3, R=0.167683293681')
ax.scatter(r_Ys[2][27][:, 0], r_Ys[2][27][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,11)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 4, R=6.25055192527e-05')
ax.scatter(r_Ys[3][20][:, 0], r_Ys[3][20][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,14)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 5, R=0.000596362331659')
ax.scatter(r_Ys[4][22][:, 0], r_Ys[4][22][:, 1], c=colors[4], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,17)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('LLE, Hole 6, R=0.00568986602902')
ax.scatter(r_Ys[5][21][:, 0], r_Ys[5][21][:, 1], c=colors[5], cmap=plt.cm.Spectral)
ax = fig.add_subplot(633)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 1, Perp=16')
ax.scatter(p_Zs[0][7][:, 0], p_Zs[0][7][:, 1], c=colors[0], cmap=plt.cm.Spectral)
ax = fig.add_subplot(636)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 2, Perp=18')
ax.scatter(p_Zs[1][8][:, 0], p_Zs[1][8][:, 1], c=colors[1], cmap=plt.cm.Spectral)
ax = fig.add_subplot(639)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 3, Perp=22')
ax.scatter(p_Zs[2][10][:, 0], p_Zs[2][10][:, 1], c=colors[2], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,12)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 4, Perp=24')
ax.scatter(p_Zs[3][11][:, 0], p_Zs[3][11][:, 1], c=colors[3], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,15)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 5, Perp=24')
ax.scatter(p_Zs[4][11][:, 0], p_Zs[4][11][:, 1], c=colors[4], cmap=plt.cm.Spectral)
ax = fig.add_subplot(6,3,18)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_title('t-SNE, Hole 6, Perp=20')
ax.scatter(p_Zs[5][9][:, 0], p_Zs[5][9][:, 1], c=colors[5], cmap=plt.cm.Spectral)
plt.tight_layout()
plt.show()

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
