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
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 10}
# A list holds the SSE values for each k
sse = []
for k in range(1, 19):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train)
    sse.append(kmeans.inertia_)

#plt.style.use("fivethirtyeight")
plt.figure()
plt.plot(range(1, 19), sse)
plt.xticks(range(1, 19))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('Sum of Squared Error (SSE) VS Number of Clusters')
plt.grid(True)
#plt.show()
plt.tight_layout()
plt.savefig('Sample_A_Part_1_kmeans_full_Sum of Squared Error (SSE) VS Number of Clusters.png',dpi=600)

kl = KneeLocator(range(1, 19), sse, curve="convex", direction="decreasing")
print('elbow position is: ',kl.elbow)

#==============================================================================
# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 19):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train)
    score = silhouette_score(x_train, kmeans.labels_)
    silhouette_coefficients.append(score)
    
#plt.style.use("fivethirtyeight")
plt.figure()
plt.plot(range(2, 19), silhouette_coefficients)
plt.xticks(range(2, 19))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.grid(True)
plt.title('Silhouette Coefficient VS Number of Clusters')
#plt.show()
plt.savefig('Sample_A_Part_1_kmeans_full_Silhouette Coefficient VS Number of Clusters.png',dpi=600)

#==============================================================================
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(1, 19), sse, label = 'Sum of Squared Error', color="C0", lw=2)
ax1.set_ylabel('Sum of Squared Error (SSE)')
ax1.set_title("Sum of Squared Error and Silhouette Coefficient VS Number of Cluster")
ax1.set_xlabel('Number of Clusters')
ax1.set_xticks(range(1, 19))
#ax1.set_ylim([83, 90])
ax1.grid()

ax2 = ax1.twinx()  # this is the important function
ax2.plot(range(2, 19), silhouette_coefficients, label = 'Silhouette Coefficient', color="C1", lw=2)
#ax2.set_xlim([0, np.e])
ax2.set_ylabel('Silhouette Coefficient')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig('Sample_A_Part_1_Sum of Squared Error and Silhouette Coefficient VS Number of Clusters.png',dpi=600)
#==============================================================================
#Pair plot
#==============================================================================
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train)
#print(x_train)
#print(kmeans.labels_)

# penguins = sns.load_dataset("penguins")
# print(penguins)

df = pd.DataFrame(x_train[:,0:5], columns = ['1','2','3','4','5'])
#df = pd.DataFrame(x_train)
df.insert(df.shape[1], 'cluster', kmeans.labels_)
print(df)
print(type(df))

sns.color_palette("hls", 8)
sns.color_palette("Set2")
#sns_plot = sns.pairplot(df, hue="cluster", markers=["o", "s", "D",'<','*'])
sns_plot = sns.pairplot(df, hue="cluster")
#ax.set(title='Pair Plot for k-means')
#plt.title('Pair Plot for k-means')
#plt.savefig('Sample_A_Part_1_Pair Plot for k-means.png',dpi=600)
sns_plot.savefig('Sample_A_Part_1_kmeans_full_Pair Plot for k-means_2.png',dpi=600)

#==============================================================================
#TSNE
#==============================================================================
tsne = manifold.TSNE(n_components=2, init='pca', random_state=10)
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
plt.title('TSNE of k-means Clustering of Sample A', y = 1.03)
plt.savefig('Sample_A_Part_1_kmeans_full_TSNE of k-means Clustering of Sample A_2.png',dpi=600)


#==============================================================================
#EM
#==============================================================================
# A list holds AIC, BIC for each k
AICs = []
BICs = []
silhouette_coefficients = []
likelihood = []
n_components = np.arange(2, 15)
for k in n_components:
    GMM = GaussianMixture(k, covariance_type='full', random_state=0)
    GMM.fit(x_train)
    AICs.append(GMM.aic(x_train))
    BICs.append(GMM.bic(x_train))
    score = silhouette_score(x_train, GMM.predict(x_train))
    silhouette_coefficients.append(score)
    likelihood.append(GMM.score(x_train))

plt.figure()
plt.plot(n_components, AICs, label='AIC')
plt.plot(n_components, BICs, label='BIC')
plt.legend(loc='best')
plt.xlabel('Number of Clusters')
plt.ylabel("Information Criterion")
plt.grid(True)
plt.title('Information Criterion VS Number of Clusters', y = 1.03)
plt.savefig('Sample_A_Part_1_EM_Information Criterion VS Number of Clusters.png',dpi=600)

plt.figure()
plt.plot(n_components, likelihood)
plt.xlabel('Number of Clusters')
plt.ylabel("Average Log-Likelihood")
plt.grid(True)
plt.title('Average Log-Likelihood VS Number of Clusters', y = 1.03)
plt.savefig('Sample_A_Part_1_EM_Average Log-Likelihood VS Number of Clusters 2.png',dpi=600)

#==============================================================================
plt.figure()
plt.plot(n_components, silhouette_coefficients)
#plt.xticks(range(2, 19))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.grid(True)
plt.title('Silhouette Coefficient VS Number of Clusters', y = 1.03)
plt.savefig('Sample_A_Part_1_EM_Silhouette Coefficient VS Number of Clusters.png',dpi=600)

#==============================================================================
#Model selection
#==============================================================================
lowest_bic = np.infty
bic_s = []
bic_t = []
bic_d = []
bic_f = []
n_components_range = range(1, 59)
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
plt.title('BIC Score for Various Model', y = 1.03)
plt.grid(True)
plt.yscale('log')
#plt.xticks(n_components_range)
plt.savefig('Sample_A_Part_1_EM_BIC Score for Various Model 4.png',dpi=600)
#===========================================================================================
GMM = GaussianMixture(10, covariance_type='full', random_state=0)
GMM.fit(x_train)

df = pd.DataFrame(x_train[:,0:5], columns = ['1','2','3','4','5'])
df.insert(df.shape[1], 'cluster', GMM.predict(x_train))
print(df)

#sns_plot = sns.pairplot(df, hue="cluster", markers=["o", "s", "D"])
sns_plot = sns.pairplot(df, hue="cluster")
sns_plot.savefig('Sample_A_Part_1_EM_Pair Plot for EM 3.png',dpi=600)
# plt.savefig('ANN_sample_A_Learning_curves_for_ANN_Default_setting.png',dpi=600)

#==============================================================================
#TSNE
#==============================================================================
tsne = manifold.TSNE(n_components=2, init='pca', random_state=10)
x_tsne = tsne.fit_transform(x_train)
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
plt.title('TSNE of EM Clustering of Sample A', y = 1.03)
plt.savefig('Sample_A_Part_1_EM_TSNE of EM Clustering of Sample A 3.png',dpi=600)

#==============================================================================
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=10)
kmeans.fit(x_train)
score_kmean = silhouette_score(x_train, kmeans.labels_)
print('silhouette score for kmean is: ',score_kmean)

GMM = GaussianMixture(3, covariance_type='full', random_state=0)
GMM.fit(x_train)
classes = GMM.predict(x_train)
score_EM = silhouette_score(x_train, classes)
print('silhouette score EM is: ',score_EM)

