# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:00:52 2021

@author: vince
"""

import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import validation_curve
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
import timeit
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn import manifold
import itertools
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FactorAnalysis

class Data():
    
    # points [1]
    def dataAllocation(self,path):
        # Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas dataframe
        data = pd.read_csv(path)
        xList = [i for i in range(data.shape[1] - 1)]
        x_data = data.iloc[:,xList]
        y_data = data.iloc[:,[-1]]
        # ------------------------------- 
        return x_data,y_data
    
    # points [1]
    def trainSets(self,x_data,y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=614, shuffle=True)       
        # -------------------------------
        return x_train, x_test, y_train, y_test


#==============================================================================
#Load data
#==============================================================================
datatest = Data()
#path = 'Class_BanknoteAuth.csv'
#path = 'pima-indians-diabetes.csv'
path = 'AFP300_nonAFP300_train_AACandDipeptide_twoSeg.csv'

x_data,y_data = datatest.dataAllocation(path)
print("dataAllocation Function Executed")

#Feature selection
#x_data = x_data.iloc[:,0:20]

x_train, x_test, y_train, y_test = datatest.trainSets(x_data,y_data)
print("trainSets Function Executed")


n = 0
for i in range(y_train.size):
    n = n + y_train.iloc[i,0]
print ('Positive rate for train data is: ',n/y_train.size)

n = 0
for i in range(y_test.size):
    n = n + y_test.iloc[i,0]
print ('Positive rate for test data is: ',n/y_test.size)

#Pre-process the data to standardize it
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#==============================================================================
#Clustering
#==============================================================================
def kmean_model_selection(x_train,n_sse,n_sil,str_see,str_sil,title_sse,title_sil):   
    kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 10}
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, n_sse):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x_train)
        sse.append(kmeans.inertia_)
    
    #plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(range(1, n_sse), sse)
    plt.xticks(range(1, n_sse))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(title_sse)
    plt.grid(True)
    #plt.show()
    plt.savefig(str_see,dpi=600)
    
    kl = KneeLocator(range(1, n_sse), sse, curve="convex", direction="decreasing")
    print('elbow position is: ',kl.elbow)
    
    #==============================================================================
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n_sil):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(x_train)
        score = silhouette_score(x_train, kmeans.labels_)
        silhouette_coefficients.append(score)
        
    #plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(range(2, n_sil), silhouette_coefficients)
    plt.xticks(range(2, n_sil))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.grid(True)
    plt.title(title_sil)
    #plt.show()
    plt.savefig(str_sil,dpi=600)
    #silhouette_coefficients = np.array(silhouette_coefficients)
    print('Max silhouettecoefficient is: ',np.max(silhouette_coefficients))

#==============================================================================
#Pair plot
#==============================================================================
def kmean_pair_plot(x_train,n_cluster,str_pair_plot):  
    kmeans = KMeans(init="random",n_clusters=n_cluster,n_init=10,max_iter=300,random_state=10)
    kmeans.fit(x_train)
    #df = pd.DataFrame(x_train, columns = ['0','1'])
    df = pd.DataFrame(x_train[:,0:4])
    df.insert(df.shape[1], 'cluster', kmeans.labels_)
    #sns_plot = sns.pairplot(df, hue="cluster", markers=["o", "s", "D",'<','*'])
    sns_plot = sns.pairplot(df, hue="cluster")
    sns_plot.savefig(str_pair_plot,dpi=600)

#==============================================================================
#TSNE
#==============================================================================
def kmean_TSNE(x_train,n_cluster,str_TSNE,title_TSNE):
    kmeans = KMeans(init="random",n_clusters=n_cluster,n_init=10,max_iter=300,random_state=10)
    kmeans.fit(x_train)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=20)
    x_tsne = tsne.fit_transform(x_train)
    classes = kmeans.labels_
    
    plt.figure()
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x_tsne[j,0] for j  in range(len(x_tsne)) if classes[j] == u]
        yi = [x_tsne[j,1] for j  in range(len(x_tsne)) if classes[j] == u]
        plt.scatter(xi, yi, color=colors[i], label=str(u), edgecolors='gray')
    plt.legend()
    plt.xlabel('t1')
    plt.ylabel("t2")
    plt.title(title_TSNE, y = 1.03)
    plt.savefig(str_TSNE,dpi=600)

#==============================================================================    
#2D plot
#==============================================================================
def kmean_2D(x_train,n_cluster,str_2D,title_2D):
    kmeans = KMeans(init="random",n_clusters=n_cluster,n_init=10,max_iter=300,random_state=10)
    kmeans.fit(x_train)
    classes = kmeans.labels_
    
    plt.figure()
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x_train[j,0] for j  in range(len(x_train)) if classes[j] == u]
        yi = [x_train[j,1] for j  in range(len(x_train)) if classes[j] == u]
        plt.scatter(xi, yi, color=colors[i], label=str(u), edgecolors='gray')
    plt.legend()
    plt.xlabel('t1')
    plt.ylabel("t2")
    plt.title(title_2D, y = 1.03)
    plt.savefig(str_2D,dpi=600)
#==============================================================================
#PCA
#==============================================================================
pca = PCA(n_components=250,random_state=0)
pca.fit(x_train)
x_train_PCA = pca.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_PCA)
x_train_PCA = scaler.transform(x_train_PCA)

n_sse = 10
n_sil = 8
str_see = 'Sample_A_Part_3_kmean_PCA_full_Sum of Squared Error (SSE) VS Number of Clusters.png'
str_sil = 'Sample_A_Part_3_kmean_PCA_full_Silhouette Coefficient VS Number of Clusters.png'
title_sse = 'Sum of Squared Error (SSE) VS Number of Clusters after PCA'
title_sil = 'Silhouette Coefficient VS Number of Clusters after PCA'
kmean_model_selection(x_train_PCA,n_sse,n_sil,str_see,str_sil,title_sse,title_sil)

str_pair_plot = 'Sample_A_Part_3_kmean_PCA_full_Pair Plot for k-means.png'
kmean_pair_plot(x_train_PCA,3,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_kmean_PCA_full_TSNE of k-means Clustering of Sample B 2.png'
title_TSNE = 'TSNE of k-means Clustering after PCA'
kmean_TSNE(x_train_PCA,3,str_TSNE,title_TSNE)

# str_2D = 'Sample_A_Part_3_kmean_PCA_2D k-means Clustering of Sample B.png'
# title_2D = 'k-means Clustering after PCA'
# kmean_2D(x_train_PCA,4,str_2D,title_2D)

#==============================================================================
#ICA
#==============================================================================
ica = FastICA(n_components=250,max_iter=2000, random_state=100)
ica.fit(x_train)
x_train_ICA = ica.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_ICA)
x_train_ICA = scaler.transform(x_train_ICA)

n_sse = 20
n_sil = 15
str_see = 'Sample_A_Part_3_kmean_ICA_full_Sum of Squared Error (SSE) VS Number of Clusters.png'
str_sil = 'Sample_A_Part_3_kmean_ICA_full_Silhouette Coefficient VS Number of Clusters.png'
title_sse = 'Sum of Squared Error (SSE) VS Number of Clusters after ICA'
title_sil = 'Silhouette Coefficient VS Number of Clusters after ICA'
kmean_model_selection(x_train_ICA,n_sse,n_sil,str_see,str_sil,title_sse,title_sil)

str_pair_plot = 'Sample_A_Part_3_kmean_ICA_Pair Plot for k-means.png'
kmean_pair_plot(x_train_ICA,3,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_kmean_ICA_full_TSNE of k-means Clustering of Sample B 2.png'
title_TSNE = 'TSNE of k-means Clustering after ICA'
kmean_TSNE(x_train_ICA,3,str_TSNE,title_TSNE)

# str_2D = 'Sample_A_Part_3_kmean_ICA_2D k-means Clustering of Sample B.png'
# title_2D = 'k-means Clustering after ICA'
# kmean_2D(x_train_ICA,3,str_2D,title_2D)

#==============================================================================
#Random projection
#==============================================================================
rp = GaussianRandomProjection(n_components=415,random_state=2)
rp.fit(x_train)
x_train_RP = rp.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_RP)
x_train_RP = scaler.transform(x_train_RP)

n_sse = 10
n_sil = 8
str_see = 'Sample_A_Part_3_kmean_RP_full_Sum of Squared Error (SSE) VS Number of Clusters.png'
str_sil = 'Sample_A_Part_3_kmean_RP_full_Silhouette Coefficient VS Number of Clusters.png'
title_sse = 'Sum of Squared Error (SSE) VS Number of Clusters after RP'
title_sil = 'Silhouette Coefficient VS Number of Clusters after RP'
kmean_model_selection(x_train_RP,n_sse,n_sil,str_see,str_sil,title_sse,title_sil)

str_pair_plot = 'Sample_A_Part_3_kmean_RP_Pair Plot for k-means.png'
kmean_pair_plot(x_train_RP,3,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_kmean_RP_TSNE of k-means Clustering of Sample B.png'
title_TSNE = 'TSNE of k-means Clustering after RP'
kmean_TSNE(x_train_RP,3,str_TSNE,title_TSNE)

# str_2D = 'Sample_B_Part_3_kmean_RP_2D k-means Clustering of Sample B.png'
# title_2D = 'k-means Clustering after RP'
# kmean_2D(x_train_RP,3,str_2D,title_2D)

#==============================================================================
#Factor analysis
#==============================================================================
fa = FactorAnalysis(n_components=300,random_state=0)
fa.fit(x_train)
x_train_FA = fa.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_FA)
x_train_FA = scaler.transform(x_train_FA)

n_sse = 20
n_sil = 15
str_see = 'Sample_A_Part_3_kmean_FA_full_Sum of Squared Error (SSE) VS Number of Clusters.png'
str_sil = 'Sample_A_Part_3_kmean_FA_full_Silhouette Coefficient VS Number of Clusters.png'
title_sse = 'Sum of Squared Error (SSE) VS Number of Clusters after FA'
title_sil = 'Silhouette Coefficient VS Number of Clusters after FA'
kmean_model_selection(x_train_FA,n_sse,n_sil,str_see,str_sil,title_sse,title_sil)

str_pair_plot = 'Sample_A_Part_3_kmean_FA_full_Pair Plot for k-means.png'
kmean_pair_plot(x_train_FA,3,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_kmean_FA_full_TSNE of k-means Clustering of Sample B.png'
title_TSNE = 'TSNE of k-means Clustering after FA'
kmean_TSNE(x_train_FA,3,str_TSNE,title_TSNE)

# str_2D = 'Sample_A_Part_3_kmean_FA_2D k-means Clustering of Sample B.png'
# title_2D = 'k-means Clustering after FA'
# kmean_2D(x_train_FA,3,str_2D,title_2D)

#==============================================================================
#EM
#==============================================================================
# A list holds AIC, BIC for each k
# AICs = []
# BICs = []
# silhouette_coefficients = []
# likelihood = []
# n_components = np.arange(2, 19)
# for k in n_components:
#     GMM = GaussianMixture(k, covariance_type='full', random_state=0)
#     GMM.fit(x_train)
#     AICs.append(GMM.aic(x_train))
#     BICs.append(GMM.bic(x_train))
#     score = silhouette_score(x_train, GMM.predict(x_train))
#     silhouette_coefficients.append(score)
#     likelihood.append(GMM.score(x_train))

# plt.figure()
# plt.plot(n_components, AICs, label='AIC')
# plt.plot(n_components, BICs, label='BIC')
# plt.legend(loc='best')
# plt.xlabel('Number of Clusters')
# plt.ylabel("Information Criterion")
# plt.grid(True)
# plt.title('Information Criterion VS Number of Clusters', y = 1.03)
# plt.savefig('Sample_B_Part_1_EM_Information Criterion VS Number of Clusters.png',dpi=600)

# plt.figure()
# plt.plot(n_components, likelihood)
# plt.xlabel('Number of Clusters')
# plt.ylabel("Average Log-Likelihood")
# plt.grid(True)
# plt.title('Average Log-Likelihood VS Number of Clusters', y = 1.03)
# plt.savefig('Sample_B_Part_1_EM_Average Log-Likelihood VS Number of Clusters.png',dpi=600)

# #==============================================================================
# plt.figure()
# plt.plot(n_components, silhouette_coefficients)
# #plt.xticks(range(2, 19))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.grid(True)
# plt.title('Silhouette Coefficient VS Number of Clusters', y = 1.03)
# plt.savefig('Sample_B_Part_1_EM_Silhouette Coefficient VS Number of Clusters.png',dpi=600)

#==============================================================================
#Model selection
#==============================================================================
def EM_model_selection(x_train,n_range,str_bic,title_bic): 
    lowest_bic = np.infty
    bic_s = []
    bic_t = []
    bic_d = []
    bic_f = []
    silhouette_coefficients = []
    n_components_range = range(1, n_range)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components,covariance_type='spherical', random_state=0)
        gmm.fit(x_train)
        bic_s.append(gmm.bic(x_train))
        
        gmm = GaussianMixture(n_components=n_components,covariance_type='tied', random_state=0)
        gmm.fit(x_train)
        bic_t.append(gmm.bic(x_train))
        
        gmm = GaussianMixture(n_components=n_components,covariance_type='diag', random_state=0)
        gmm.fit(x_train)
        bic_d.append(gmm.bic(x_train))
        
        gmm = GaussianMixture(n_components=n_components,covariance_type='full', random_state=0)
        gmm.fit(x_train)
        bic_f.append(gmm.bic(x_train))
    
    # Plot the BIC scores
    plt.figure()
    plt.plot(n_components_range,bic_s, label = 'spherical', color="C0", lw=2)
    plt.plot(n_components_range,bic_t, label = 'tied', color="C1", lw=2)
    plt.plot(n_components_range,bic_d, label = 'diag', color="C2", lw=2)
    plt.plot(n_components_range,bic_f, label = 'full', color="C3", lw=2)
    plt.legend()
    plt.xlabel('Number of components')
    plt.ylabel("BIC")
    plt.title(title_bic, y = 1.03)
    plt.grid(True)
    plt.xticks(n_components_range)
    plt.savefig(str_bic,dpi=600)
#===========================================================================================
def EM_pair_plot(x_train,n_cluster,c_type,str_pair_plot): 
    GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
    GMM.fit(x_train)
    
    df = pd.DataFrame(x_train[:,0:4])
    df.insert(df.shape[1], 'cluster', GMM.predict(x_train))
    print(df)
    
    sns_plot = sns.pairplot(df, hue="cluster")
    sns_plot.savefig(str_pair_plot,dpi=600)


#==============================================================================
#TSNE
#==============================================================================
def EM_TSNE(x_train,n_cluster,c_type,str_TSNE,title_TSNE): 
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=20)
    x_tsne = tsne.fit_transform(x_train)
    GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
    GMM.fit(x_train)
    classes = GMM.predict(x_train)
    
    plt.figure()
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x_tsne[j,0] for j  in range(len(x_tsne)) if classes[j] == u]
        yi = [x_tsne[j,1] for j  in range(len(x_tsne)) if classes[j] == u]
        plt.scatter(xi, yi, color=colors[i], label=str(u),edgecolors='gray')
    plt.legend()
    plt.xlabel('t1')
    plt.ylabel("t2")
    plt.title(title_TSNE, y = 1.03)
    plt.savefig(str_TSNE,dpi=600)
    
#==============================================================================
#2D plot
#==============================================================================
def EM_2d(x_train,n_cluster,c_type,str_2d,title_2d): 
    GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
    GMM.fit(x_train)
    classes = GMM.predict(x_train)
    
    plt.figure()
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x_train[j,0] for j  in range(len(x_train)) if classes[j] == u]
        yi = [x_train[j,1] for j  in range(len(x_train)) if classes[j] == u]
        plt.scatter(xi, yi, color=colors[i], label=str(u),edgecolors='gray')
    plt.legend()
    plt.xlabel('t1')
    plt.ylabel("t2")
    plt.title(title_2d, y = 1.03)
    plt.savefig(str_2d,dpi=600)

#==============================================================================
#PCA
#==============================================================================
pca = PCA(n_components=250,random_state=0)
pca.fit(x_train)
x_train_PCA = pca.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_PCA)
x_train_PCA = scaler.transform(x_train_PCA)


str_bic = 'Sample_A_Part_3_EM_PCA_full_BIC Score for Various Model.png'
title_bic = 'BIC Score for Various Model after PCA'
EM_model_selection(x_train_PCA,8,str_bic,title_bic)

c_type = 'diag'
n_cluster = 4

str_pair_plot = 'Sample_A_Part_3_EM_PCA_full_Pair Plot for EM.png'
EM_pair_plot(x_train_PCA,n_cluster,c_type,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_EM_PCA_full_TSNE of EM Clustering of Sample A.png'
title_TSNE = 'TSNE of EM Clustering after PCA'
EM_TSNE(x_train_PCA,n_cluster,c_type,str_TSNE,title_TSNE)

# str_2d = 'Sample_A_Part_3_EM_PCA_EM Clustering of Sample B.png'
# title_2d = 'EM Clustering after PCA'
# EM_2d(x_train_PCA,n_cluster,c_type,str_2d,title_2d)


GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
GMM.fit(x_train_PCA)
classes = GMM.predict(x_train_PCA)
score = silhouette_score(x_train_PCA, classes)
print('Max silhouettecoefficient is: ',score)

#==============================================================================
#ICA
#==============================================================================
ica = FastICA(n_components=250,max_iter=2000, random_state=100)
ica.fit(x_train)
x_train_ICA = ica.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_ICA)
x_train_ICA = scaler.transform(x_train_ICA)

str_bic = 'Sample_A_Part_3_EM_ICA_full_BIC Score for Various Model.png'
title_bic = 'BIC Score for Various Model after ICA'
EM_model_selection(x_train_ICA,8,str_bic,title_bic)

c_type = 'diag'
n_cluster = 5

str_pair_plot = 'Sample_A_Part_3_EM_ICA_full_Pair Plot for EM.png'
EM_pair_plot(x_train_ICA,n_cluster,c_type,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_EM_ICA_full_TSNE of EM Clustering of Sample B.png'
title_TSNE = 'TSNE of EM Clustering after ICA'
EM_TSNE(x_train_ICA,n_cluster,c_type,str_TSNE,title_TSNE)

# str_2d = 'Sample_A_Part_3_EM_ICA_EM Clustering of Sample B.png'
# title_2d = 'EM Clustering after ICA'
# EM_2d(x_train_ICA,n_cluster,c_type,str_2d,title_2d)


GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
GMM.fit(x_train_ICA)
classes = GMM.predict(x_train_ICA)
score = silhouette_score(x_train_ICA, classes)
print('Max silhouettecoefficient is: ',score)

#==============================================================================
#Random projection
#==============================================================================
rp = GaussianRandomProjection(n_components=415,random_state=2)
rp.fit(x_train)
x_train_RP = rp.transform(x_train)

str_bic = 'Sample_A_Part_3_EM_RP_full_BIC Score for Various Model.png'
title_bic = 'BIC Score for Various Model after RP'
EM_model_selection(x_train_RP,20,str_bic,title_bic)

c_type = 'diag'
n_cluster = 4

str_pair_plot = 'Sample_A_Part_3_EM_RP_full_Pair Plot for EM.png'
EM_pair_plot(x_train_RP,n_cluster,c_type,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_EM_RP_full_TSNE of EM Clustering of Sample B.png'
title_TSNE = 'TSNE of EM Clustering after RP'
EM_TSNE(x_train_RP,n_cluster,c_type,str_TSNE,title_TSNE)

# str_2d = 'Sample_B_Part_3_EM_RP_EM Clustering of Sample B.png'
# title_2d = 'EM Clustering after RP'
# EM_2d(x_train_RP,n_cluster,c_type,str_2d,title_2d)


GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
GMM.fit(x_train_RP)
classes = GMM.predict(x_train_RP)
score = silhouette_score(x_train_RP, classes)
print('Max silhouettecoefficient is: ',score)

#==============================================================================
#Factor analysis
#==============================================================================
fa = FactorAnalysis(n_components=300,random_state=0)
fa.fit(x_train)
x_train_FA = fa.transform(x_train)
scaler = StandardScaler()
scaler.fit(x_train_FA)
x_train_FA = scaler.transform(x_train_FA)

str_bic = 'Sample_A_Part_3_EM_FA_BIC Score for Various Model.png'
title_bic = 'BIC Score for Various Model after FA'
EM_model_selection(x_train_FA,8,str_bic,title_bic)

c_type = 'diag'
n_cluster = 4

str_pair_plot = 'Sample_A_Part_3_EM_FA_full_Pair Plot for EM.png'
EM_pair_plot(x_train_FA,n_cluster,c_type,str_pair_plot)

str_TSNE = 'Sample_A_Part_3_EM_FA_full_TSNE of EM Clustering of Sample B.png'
title_TSNE = 'TSNE of EM Clustering after FA'
EM_TSNE(x_train_FA,n_cluster,c_type,str_TSNE,title_TSNE)

# str_2d = 'Sample_A_Part_3_EM_FA_EM Clustering of Sample B.png'
# title_2d = 'EM Clustering after FA'
# EM_2d(x_train_FA,n_cluster,c_type,str_2d,title_2d)


GMM = GaussianMixture(n_cluster, covariance_type=c_type, random_state=0)
GMM.fit(x_train_FA)
classes = GMM.predict(x_train_FA)
score = silhouette_score(x_train_FA, classes)
print('Max silhouettecoefficient is: ',score)



