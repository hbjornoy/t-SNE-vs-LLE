import numpy as np
import pickle
import time
import helpers as HL
from sklearn.manifold import TSNE as t_sne
from sklearn.manifold import LocallyLinearEmbedding as lle

def get_swiss_roll(method,folder, create=False, n=1000, noise=0.01):
    if not (method=='tsne' or method=='lle'):
        print('Not valid method')
    if create: 
        X, color, X_2d=HL.make_swissroll(n=n, noise=noise)
        pickle.dump( X, open(folder+"/X_"+method+".pkl", "wb")) 
        pickle.dump( color, open(folder+"/color_"+method+".pkl", "wb")) 
        pickle.dump( X_2d, open(folder+"/X_2d_"+method+".pkl", "wb")) 
    else: 
        color=pickle.load(open(folder+"/color_"+method+".pkl", "rb"))
        X=pickle.load(open(folder+"/X_"+method+".pkl", "rb"))
        X_2d=pickle.load(open(folder+"/X_2d_"+method+".pkl", "rb"))
    return color,X,X_2d
def perpexity(folder,per=np.arange(2,150,2), create=False): 
    if create: 
        p_Z=[]
        p_times=np.zeros(len(per))
        p_kl_divergence=np.zeros(len(per))
        X=pickle.load(open(folder+"/X_tsne.pkl", "rb"))
        for i, p in enumerate(per):
            tsne=t_sne(perplexity=p)
            start_time=time.time()
            p_Z.append(tsne.fit_transform(X))
            p_times[i]= time.time()-start_time
            p_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( p_Z, open(folder+"/p_Z_tsne.pkl", "wb")) 
        pickle.dump(per, open(folder+"/per.pkl","wb"))
        pickle.dump(p_times, open(folder+"/p_times.pkl","wb"))
        pickle.dump(p_kl_divergence, open(folder+"/p_kl_divergence.pkl","wb"))
    else: 
        p_Z= pickle.load(open(folder+"/p_Z_tsne.pkl", "rb"))
        per=pickle.load(open(folder+"/per.pkl", "rb"))
        p_times=pickle.load(open(folder+"/p_times.pkl", "rb"))
        p_kl_divergence=pickle.load(open(folder+"/p_kl_divergence.pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d_tsne.pkl", "rb"))
    p_differences=HL.get_differences(X_2d_tsne,p_Z)
    return p_Z,per,p_times,p_kl_divergence,p_differences

def early_exxaggeration(folder, create=False,early_exaggeration=np.arange(1,80,1)): 
    if create: 
        e_Z=[]
        e_times=np.zeros(len(early_exaggeration))
        e_kl_divergence=np.zeros(len(early_exaggeration))
        X=pickle.load(open(folder+"/X_tsne.pkl", "rb"))
        for i, e in enumerate(early_exaggeration):
            tsne=t_sne(early_exaggeration=e)
            start_time=time.time()
            e_Z.append(tsne.fit_transform(X))
            e_times[i]= time.time()-start_time
            e_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( e_Z, open(folder+"/e_Z_tsne.pkl", "wb")) 
        pickle.dump(early_exaggeration, open(folder+"/early_exaggeration.pkl","wb"))
        pickle.dump(e_times, open(folder+"/e_times.pkl","wb"))
        pickle.dump(e_kl_divergence, open(folder+"/e_kl_divergence.pkl","wb"))
    else: 
        e_Z= pickle.load(open(folder+"/e_Z_tsne.pkl", "rb"))
        early_exaggeration=pickle.load(open(folder+"/early_exaggeration.pkl", "rb"))
        e_times=pickle.load(open(folder+"/e_times.pkl", "rb"))
        e_kl_divergence=pickle.load(open(folder+"/e_kl_divergence.pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d_tsne.pkl", "rb"))
    e_differences=HL.get_differences(X_2d_tsne,e_Z)
    return e_Z,early_exaggeration,e_times,e_kl_divergence,e_differences

def learning_rates(folder, create=False, learning_rates=np.arange(5,1000,5) ): 
    if create: 
        l_Z=[]
        l_times=np.zeros(len(learning_rates))
        l_kl_divergence=np.zeros(len(learning_rates))
        X=pickle.load(open(folder+"/X_tsne.pkl", "rb"))
        for i, l in enumerate(learning_rates):
            tsne=t_sne(learning_rate=l)
            start_time=time.time()
            l_Z.append(tsne.fit_transform(X))
            l_times[i]= time.time()-start_time
            l_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( l_Z, open(folder+"/l_Z_tsne.pkl", "wb")) 
        pickle.dump(learning_rates, open(folder+"/learning_rates.pkl","wb"))
        pickle.dump(l_times, open(folder+"/l_times.pkl","wb"))
        pickle.dump(l_kl_divergence, open(folder+"/l_kl_divergence.pkl","wb"))
    else: 
        l_Z= pickle.load(open(folder+"/l_Z_tsne.pkl", "rb"))
        learning_rates=pickle.load(open(folder+"/learning_rates.pkl", "rb"))
        l_times=pickle.load(open(folder+"/l_times.pkl", "rb"))
        l_kl_divergence=pickle.load(open(folder+"/l_kl_divergence.pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d_tsne.pkl", "rb"))
    l_differences=HL.get_differences(X_2d_tsne,l_Z)
    return l_Z,learning_rates,l_times,l_kl_divergence,l_differences


def threshold(folder, create=False,threshold=np.logspace(-14,-1,50) ): 
    if create: 
        t_Z=[]
        t_times=np.zeros(len(threshold))
        t_kl_divergence=np.zeros(len(threshold))
        X=pickle.load(open(folder+"/X_tsne.pkl", "rb"))
        for i, t in enumerate(threshold):
            tsne=t_sne(min_grad_norm=t)
            start_time=time.time()
            t_Z.append(tsne.fit_transform(X))
            t_times[i]= time.time()-start_time
            t_kl_divergence[i]=tsne.kl_divergence_
        pickle.dump( t_Z, open(folder+"/t_Z_tsne.pkl", "wb")) 
        pickle.dump(threshold, open(folder+"/threshold.pkl","wb"))
        pickle.dump(t_times, open(folder+"/t_times.pkl","wb"))
        pickle.dump(t_kl_divergence, open(folder+"/t_kl_divergence.pkl","wb"))
    else: 
        t_Z= pickle.load(open(folder+"/t_Z_tsne.pkl", "rb"))
        threshold=pickle.load(open(folder+"/threshold.pkl", "rb"))
        t_times=pickle.load(open(folder+"/t_times.pkl", "rb"))
        t_kl_divergence=pickle.load(open(folder+"/t_kl_divergence.pkl", "rb"))
    X_2d_tsne=pickle.load(open(folder+"/X_2d_tsne.pkl", "rb"))
    t_differences=HL.get_differences(X_2d_tsne,t_Z)
    return t_Z,threshold,t_times,t_kl_divergence,t_differences


def n_neighbors(folder, create=False,n_neighbors=np.arange(3,60,1)):  
    if create: 
        n_components=2
        X=pickle.load(open(folder+"/X_lle.pkl", "rb"))
        n_Y=[]
        n_times=np.zeros(len(n_neighbors))
        n_reconstruction_error=np.zeros(len(n_neighbors))
        for i, n in enumerate(n_neighbors):
            LLE=lle(n, n_components,eigen_solver='auto')
            start_time=time.time()
            n_Y.append(LLE.fit_transform(X))
            n_times[i]= time.time()-start_time
            n_reconstruction_error[i]=LLE.reconstruction_error_ 
        pickle.dump( n_Y, open(folder+"/n_Y_lle.pkl", "wb")) 
        #pickle.dump(lle_color, open("lle_color.pkl","wb"))
        pickle.dump(n_neighbors, open(folder+"/n_neighbors.pkl","wb"))
        pickle.dump(n_times, open(folder+"/n_times.pkl","wb"))
        pickle.dump(n_reconstruction_error, open(folder+"/n_reconstruction_error.pkl","wb"))
    else: 
        n_Y= pickle.load(open(folder+"/n_Y_lle.pkl", "rb"))
        #lle_color=pickle.load(open("lle_color.pkl", "rb"))
        n_neighbors=pickle.load(open(folder+"/n_neighbors.pkl", "rb"))
        n_times=pickle.load(open(folder+"/n_times.pkl", "rb"))
        n_reconstruction_error=pickle.load(open(folder+"/n_reconstruction_error.pkl", "rb"))
    X_2d_lle=pickle.load(open(folder+"/X_2d_lle.pkl", "rb"))
    n_differences=HL.get_differences(X_2d_lle,n_Y)
    return n_Y,n_neighbors, n_times,n_reconstruction_error,n_differences

def n_reg(folder, create=False,reg=np.logspace(-14,10,50)): 
    if create:
        X=pickle.load(open(folder+"/X_lle.pkl", "rb"))
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
        pickle.dump( r_Y, open(folder+"/r_Y_lle.pkl", "wb")) 
        #pickle.dump(lle_color, open("lle_color.pkl","wb"))
        pickle.dump(reg, open(folder+"/reg.pkl","wb"))
        pickle.dump(r_times, open(folder+"/r_times.pkl","wb"))
        pickle.dump(r_reconstruction_error, open(folder+"/r_reconstruction_error.pkl","wb"))
    else: 
        r_Y= pickle.load(open(folder+"/r_Y_lle.pkl", "rb"))
        #lle_color=pickle.load(open("lle_color.pkl", "rb"))
        reg=pickle.load(open(folder+"/reg.pkl", "rb"))
        r_times=pickle.load(open(folder+"/r_times.pkl", "rb"))
        r_reconstruction_error=pickle.load(open(folder+"/r_reconstruction_error.pkl", "rb"))
    X_2d_lle=pickle.load(open(folder+"/X_2d_lle.pkl", "rb"))
    r_differences=HL.get_differences(X_2d_lle,r_Y)
    return r_Y,reg, r_times,r_reconstruction_error,r_differences