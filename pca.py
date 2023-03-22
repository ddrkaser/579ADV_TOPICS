import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.manifold import MDS

face_dataset = fetch_olivetti_faces()
face_images = face_dataset.images
face_data = face_dataset.data
face_data_df = pd.DataFrame(face_data)
face_data_df[100].describe()
face_target = face_dataset.target

cov_data = np.corrcoef(face_data.T)
cov_data

#standardization
scaler = preprocessing.StandardScaler()
face_data_standardized = scaler.fit_transform(face_data)
face_data_standardized[:,:5].mean(axis = 0)
face_data_standardized[:,:5].std(axis = 0)

#load data
face_dataset = fetch_olivetti_faces()
face_data = face_dataset.data
face_target = face_dataset.target
face_data_standardized = preprocessing.StandardScaler().fit_transform(face_data)


def pca(data,k, explain_ratio = False):
    #compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    #compute eigen values and corresponding eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    #sort eigen values in indeces and flip it into descending order
    sorted_components = np.argsort(eigen_values)
    sorted_components = np.flip(sorted_components)
    #pick top k eigen vectors as feature vectors
    feature_vectors = eigen_vectors[:,sorted_components[:k]]
    if explain_ratio:
        explained_variance = eigen_values[sorted_components]
        explained_variance_ratio = explained_variance / eigen_values.sum()
        return np.dot(data, feature_vectors), feature_vectors, explained_variance_ratio
    return np.dot(data, feature_vectors), feature_vectors

def inverse_transform(data, feature_vectors):
    return np.dot(data, feature_vectors.T)

pca_data, feature_vectors, explained_variance_ratio = pca(face_data, 2, explain_ratio = True)
plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c = face_target)

#choose k, 200 can explain about 0.95 data
plt.plot(np.cumsum(explained_variance_ratio[:300]))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

pca_data, feature_vectors = pca(face_data, 200)
inversed_pca = inverse_transform(pca_data, feature_vectors)

avg_diff = abs(face_data_standardized - inversed_pca).mean(axis = 1)
avg_diff.mean()
mse = ((face_data_standardized - inversed_pca)**2).mean(axis = 1)
avg_mse = mse.mean().real
lin_mse = mean_squared_error(face_data_standardized, inversed_pca.real)


dm_face_data = pairwise_distances(face_data)
dm_pca_data = pairwise_distances(pca_data.real)
dm_std_face_data = pairwise_distances(face_data_standardized)
ee = distance.pdist(pca_data.real, 'euclidean')
ff = distance.pdist(face_data_standardized, 'euclidean')

aa = dm_face_data/dm_pca_data
bb = dm_std_face_data/dm_pca_data
cc = dm_std_face_data - dm_pca_data
dd = distance.pdist(face_data_standardized, 'euclidean') - distance.pdist(pca_data.real, 'euclidean')
dm_mse = mean_squared_error(ee, ff)

np.linalg.norm(dm_std_face_data-dm_pca_data)

mds = MDS(n_components=200)
mds_data = mds.fit_transform(face_data_standardized)
dm_mds_data = pairwise_distances(mds_data)
jj = distance.pdist(mds_data, 'euclidean')
dm_mse_mds = mean_squared_error(jj, ff)

class convers_pca():
    def __init__(self, no_of_components):
        self.no_of_components = no_of_components
        self.eigen_values = None
        self.eigen_vectors = None

    def transform(self, data):
        return np.dot(data - self.mean, self.projection_matrix.T)

    def inverse_transform(self, data):
        return np.dot(data, self.projection_matrix) + self.mean

    def fit(self, x):
        self.no_of_components = x.shape[1]
        self.mean = np.mean(x, axis=0)

        cov_matrix = np.cov(x - self.mean, rowvar=False)

        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        self.eigen_vectors = self.eigen_vectors.T

        self.sorted_components = np.argsort(self.eigen_values)[::-1]

        self.projection_matrix = self.eigen_vectors[self.sorted_components[:self.no_of_components]]
        self.explained_variance = self.eigen_values[self.sorted_components]
        self.explained_variance_ratio = self.explained_variance / self.eigen_values.sum()
