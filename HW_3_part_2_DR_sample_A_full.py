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
#PCA
#==============================================================================
n = np.min([x_train.shape[0],x_train.shape[1]])
pca = PCA(n_components=n,random_state=0)
pca.fit(x_train)
#X_r_PCA = pca.fit(x_train).transform(x_train)

n_component_range = range(1,n+1)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(n_component_range, pca.explained_variance_ratio_,label='Varaince Ratio', color="C0", lw=2)
ax1.set_ylabel('Percentage of Variance Explained')
ax1.set_title("Variance VS Components for PCA")
ax1.set_xlabel('Components')
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(n_component_range, np.cumsum(pca.explained_variance_ratio_),label='Accumulated Varaince Ratio', color="C1", lw=2)
ax2.set_ylabel('Accumulated Variance Ratio')
fig.legend(loc="upper right", bbox_to_anchor=(0.6,0.2), bbox_transform=ax1.transAxes)
plt.tight_layout()
#plt.xticks(n_component_range)
plt.savefig('Sample_A_Part_2_full_Variance VS Components for PCA.png',dpi=600)
#==============================================================================
n = np.min([x_train.shape[0],x_train.shape[1]])
MSE_PCA = []
for i in range(1,n+1):    
    pca = PCA(n_components=i,random_state=0)
    pca.fit(x_train)
    x_train_transform_PCA = pca.transform(x_train)
    x_train_reconstruct = pca.inverse_transform(x_train_transform_PCA[:,0:i])
    MSE = np.mean((x_train - x_train_reconstruct)**2)
    MSE_PCA.append(MSE)

plt.figure()
plt.plot(range(1,n+1), MSE_PCA)
#plt.xticks(range(1,n+1))
plt.xlabel("Number of Components")
plt.ylabel("Mean Square Error (MSE)")
plt.title('MSE VS Number of Components for PCA')
plt.grid(True)
plt.savefig('Sample_A_Part_2_full_MSE VS Number of Components for PCA.png',dpi=600)

# x_train_transform_PCA = pca.transform(x_train)
# inverse_data = np.linalg.pinv(pca.components_.T)
# reconstructed_data = x_train_transform_PCA.dot(inverse_data)
# MSE = np.mean((x_train - reconstructed_data)**2)
# print(MSE)

#==============================================================================
#ICA
#==============================================================================

# def Kurt(x):
#     n = np.shape(x)[0]
#     mean = np.sum((x**1)/n) # Calculate the mean
#     var = np.sum((x-mean)**2)/n # Calculate the variance
#     skew = np.sum((x-mean)**3)/n # Calculate the skewness
#     kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
#     kurt = kurt/(var**2)-3
#     return kurt
n_comp = np.min([x_train.shape[0],x_train.shape[1]])
kurts = []
for k in range(1,n_comp-5,10):

    ica = FastICA(n_components=k,max_iter=1000, random_state=100)
    ica.fit(x_train)
    x_train_transform_ICA = ica.transform(x_train)
    n = x_train_transform_ICA.shape[1]
    kurt = 0
    for i in range(n):
        kurt = kurt + kurtosis(x_train_transform_ICA[:,i])
    kurt = kurt/n
    kurts.append(kurt)


plt.figure()
plt.plot(range(1,n_comp-5,10), kurts)
#plt.xticks(range(1,n_comp-5,10))
plt.xlabel("Number of Components")
plt.ylabel("Avrage Kurtosis")
plt.title('Avrage Kurtosis VS Number of Components for ICA')
plt.grid(True)
plt.savefig('Sample_A_Part_2_full_Avrage Kurtosis VS Number of Components for ICA 2.png',dpi=600)  

#=========================================================================================
MSE_mean_ICA = []
MSE_std_ICA = []
n_comp = np.min([x_train.shape[0],x_train.shape[1]])
for k in range(1,n_comp-5,10):
    MSEs = []
    for i in range(0,1):   
        ica = FastICA(n_components=k,max_iter=2000,random_state=i*10)
        x_train_transform_ICA = ica.fit_transform(x_train)
        inverse_data = np.linalg.pinv(ica.components_.T)
        reconstructed_data = x_train_transform_ICA.dot(inverse_data)
        MSE = np.mean((x_train - reconstructed_data)**2)
        MSEs.append(MSE)
    MSE_mean_ICA.append(np.mean(MSEs, axis=0))
    MSE_std_ICA.append(np.std(MSEs, axis=0))

MSE_mean_ICA = np.array(MSE_mean_ICA)
MSE_std_ICA = np.array(MSE_std_ICA)

plt.figure()
plt.plot(range(1,n_comp-5,10),MSE_mean_ICA, label = 'MSE for ICA', color="C0", lw=2)
plt.fill_between(range(1,n_comp-5,10), MSE_mean_ICA - MSE_std_ICA, MSE_mean_ICA + MSE_std_ICA, alpha=0.2,color="C0", lw=2)
plt.ylabel('Mean Square Error (MSE)', fontsize = 14)
plt.xlabel('Number of Components', fontsize = 14)
plt.title('MSE VS Number of Components for ICA', y = 1.03)
#plt.legend(loc=0)
plt.grid(True)
#plt.xticks(range(1,n_comp-5,10))
plt.tight_layout()
plt.savefig('Sample_A_Part_2_full_MSE VS Number of Components for ICA.png',dpi=600)

#==============================================================================
#Random projection
#==============================================================================
MSE_mean_RP = []
MSE_std_RP = []
for k in range(1,x_train.shape[1]+1,10):
    MSEs = []
    for i in range(0,10):   
        rp = GaussianRandomProjection(n_components=k,random_state=i)
        x_train_transform_RP = rp.fit_transform(x_train)
        inverse_data = np.linalg.pinv(rp.components_.T)
        reconstructed_data = x_train_transform_RP.dot(inverse_data)
        MSE = np.mean((x_train - reconstructed_data)**2)
        MSEs.append(MSE)
    MSE_mean_RP.append(np.mean(MSEs, axis=0))
    MSE_std_RP.append(np.std(MSEs, axis=0))

MSE_mean_RP = np.array(MSE_mean_RP)
MSE_std_RP = np.array(MSE_std_RP)

plt.figure()
plt.plot(range(1,x_train.shape[1]+1,10),MSE_mean_RP, label = 'MSE for RP', color="C0", lw=2)
plt.fill_between(range(1,x_train.shape[1]+1,10), MSE_mean_RP - MSE_std_RP, MSE_mean_RP + MSE_std_RP, alpha=0.2,color="C0", lw=2)
plt.ylabel('Mean Square Error (MSE)', fontsize = 14)
plt.xlabel('Number of Components', fontsize = 14)
plt.title('MSE VS Number of Components for Random Projection', y = 1.03)
#plt.legend(loc=0)
plt.grid(True)
#plt.xticks(range(1,x_train.shape[1]+1))
plt.tight_layout()
plt.savefig('Sample_A_Part_2_full_MSE VS Number of Components for Random Projection.png',dpi=600)

#==============================================================================
#Factor Analysis
#==============================================================================
MSE_mean_FA = []
MSE_std_FA = []
n_comp = np.min([x_train.shape[0],x_train.shape[1]])
for k in range(1,x_train.shape[1]+1,10):
    MSEs = []
    for i in range(0,2):   
        fa = FactorAnalysis(n_components=k,random_state=i)
        x_train_transform_FA = fa.fit_transform(x_train)
        inverse_data = np.linalg.pinv(fa.components_.T)
        reconstructed_data = x_train_transform_FA.dot(inverse_data)
        MSE = np.mean((x_train - reconstructed_data)**2)
        MSEs.append(MSE)
    MSE_mean_FA.append(np.mean(MSEs, axis=0))
    MSE_std_FA.append(np.std(MSEs, axis=0))
    print(k)

MSE_mean_FA = np.array(MSE_mean_FA)
MSE_std_FA = np.array(MSE_std_FA)

plt.figure()
plt.plot(range(1,x_train.shape[1]+1,10),MSE_mean_FA, label = 'MSE for FA', color="C0", lw=2)
plt.fill_between(range(1,x_train.shape[1]+1,10), MSE_mean_FA - MSE_std_FA, MSE_mean_FA + MSE_std_FA, alpha=0.2,color="C0", lw=2)
plt.ylabel('Mean Square Error (MSE)', fontsize = 14)
plt.xlabel('Number of Components', fontsize = 14)
plt.title('MSE VS Number of Components for Factor Analysis', y = 1.03)
#plt.legend(loc=0)
plt.grid(True)
#plt.xticks(range(1,x_train.shape[1]+1))
plt.tight_layout()
plt.savefig('Sample_A_Part_2_full_MSE VS Number of Components for Factor Analysis.png',dpi=600)

#==============================================================================
#Reconstructuion error
#==============================================================================
n_comp = np.min([x_train.shape[0],x_train.shape[1]])
plt.figure()
plt.plot(range(1,n_comp+1),MSE_PCA, label = 'MSE for PCA', color="C0", lw=2)
#plt.plot(range(1,n_comp-5,10),MSE_mean_ICA, label = 'MSE for ICA', color="C1", lw=2)
#plt.fill_between(range(1,n_comp-5,10), MSE_mean_ICA - MSE_std_ICA, MSE_mean_ICA + MSE_std_ICA, alpha=0.2,color="C1", lw=2)
plt.plot(range(1,x_train.shape[1]+1,10),MSE_mean_RP, label = 'MSE for RP', color="C2", lw=2)
plt.fill_between(range(1,x_train.shape[1]+1,10), MSE_mean_RP - MSE_std_RP, MSE_mean_RP + MSE_std_RP, alpha=0.2,color="C2", lw=2)
plt.plot(range(1,x_train.shape[1]+1,10),MSE_mean_FA, label = 'MSE for FA', color="C3", lw=2)
plt.fill_between(range(1,x_train.shape[1]+1,10), MSE_mean_FA - MSE_std_FA, MSE_mean_FA + MSE_std_FA, alpha=0.2,color="C3", lw=2)
plt.ylabel('Mean Square Error (MSE)', fontsize = 14)
plt.xlabel('Number of Components', fontsize = 14)
plt.title('MSE VS Number of Components for Dimensionality Reduction', y = 1.03)
plt.legend(loc=0)
plt.grid(True)
#plt.xticks(range(1,x_train.shape[1]+1))
plt.tight_layout()
plt.savefig('Sample_A_Part_2_full_MSE VS Number of Components for Dimensionality Reduction 2.png',dpi=600)


#==============================================================================
#time
#==============================================================================
start_kmeans = time.time()
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train)
end_kmeans = time.time()
print('Time for kmeans clustering is: ',end_kmeans-start_kmeans)

start_EM = time.time()
GMM = GaussianMixture(3, covariance_type='tied', random_state=0)
GMM.fit(x_train)
end_EM = time.time()
print('Time for EM clustering is: ',end_EM-start_EM)
#==============================================================================
#PCA
#==============================================================================
start_PCA = time.time()
pca = PCA(n_components=250,random_state=0)
pca.fit(x_train)
x_train_transform_PCA = pca.transform(x_train)
end_PCA = time.time()
x_train_reconstruct = pca.inverse_transform(x_train_transform_PCA)
MSE_PCA = np.mean((x_train - x_train_reconstruct)**2)

print('Accumulated Variance Ratio is: ',np.sum(pca.explained_variance_ratio_))
print('MSE for PCA is: ',MSE_PCA)
print('Time for PCA is: ',end_PCA-start_PCA)

start_kmeans_PCA = time.time()
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train_transform_PCA)
end_kmeans_PCA = time.time()
print('Time for PCA kmeans clustering is: ',end_kmeans_PCA-start_kmeans_PCA)

start_EM_PCA = time.time()
GMM = GaussianMixture(4, covariance_type='diag', random_state=0)
GMM.fit(x_train_transform_PCA)
end_EM_PCA = time.time()
print('Time for PCA EM clustering is: ',end_EM_PCA-start_EM_PCA)

#==============================================================================
#ICA
#==============================================================================
start_ICA = time.time()
ica = FastICA(n_components=250,max_iter=2000, random_state=100)
ica.fit(x_train)
x_train_transform_ICA = ica.transform(x_train)
end_ICA = time.time()
n = x_train_transform_ICA.shape[1]
kurt = 0
for i in range(n):
    kurt = kurt + kurtosis(x_train_transform_ICA[:,i])
kurt = kurt/n

inverse_data = np.linalg.pinv(ica.components_.T)
reconstructed_data = x_train_transform_ICA.dot(inverse_data)
MSE_ICA = np.mean((x_train - reconstructed_data)**2)

print('Average kurtosis is: ',kurt)
print('MSE for ICA is: ',MSE_ICA)
print('Time for ICA is: ',end_ICA-start_ICA)

start_kmeans_ICA = time.time()
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train_transform_ICA)
end_kmeans_ICA = time.time()
print('Time for ICA kmeans clustering is: ',end_kmeans_ICA-start_kmeans_ICA)

start_EM_ICA = time.time()
GMM = GaussianMixture(5, covariance_type='diag', random_state=0)
GMM.fit(x_train_transform_ICA)
end_EM_ICA = time.time()
print('Time for ICA EM clustering is: ',end_EM_ICA-start_EM_ICA)

#==============================================================================
#Random projection
#==============================================================================
start_RP = time.time()
rp = GaussianRandomProjection(n_components=415,random_state=2)
x_train_transform_RP = rp.fit_transform(x_train)
end_RP = time.time()
inverse_data = np.linalg.pinv(rp.components_.T)
reconstructed_data = x_train_transform_RP.dot(inverse_data)
MSE_RP = np.mean((x_train - reconstructed_data)**2)

print('MSE for RP is: ',MSE_RP)
print('Time for RP is: ',end_RP-start_RP)

start_kmeans_RP = time.time()
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train_transform_RP)
end_kmeans_RP = time.time()
print('Time for RP kmeans clustering is: ',end_kmeans_RP-start_kmeans_RP)

start_EM_RP = time.time()
GMM = GaussianMixture(4, covariance_type='diag', random_state=0)
GMM.fit(x_train_transform_RP)
end_EM_RP = time.time()
print('Time for RP EM clustering is: ',end_EM_RP-start_EM_RP)

#==============================================================================
#Factor analysis
#==============================================================================
start_FA = time.time()
fa = FactorAnalysis(n_components=300,random_state=0)
x_train_transform_FA = fa.fit_transform(x_train)
end_FA = time.time()
inverse_data = np.linalg.pinv(fa.components_.T)
reconstructed_data = x_train_transform_FA.dot(inverse_data)
MSE_FA = np.mean((x_train - reconstructed_data)**2)

print('MSE for FA is: ',MSE_FA)
print('Time for FA is: ',end_FA-start_FA)

start_kmeans_FA = time.time()
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train_transform_FA)
end_kmeans_FA = time.time()
print('Time for FA kmeans clustering is: ',end_kmeans_FA-start_kmeans_FA)

start_EM_FA = time.time()
GMM = GaussianMixture(4, covariance_type='diag', random_state=0)
GMM.fit(x_train_transform_FA)
end_EM_FA = time.time()
print('Time for FA EM clustering is: ',end_EM_FA-start_EM_FA)


