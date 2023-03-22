# Download the faces dataset and normalize or standardize it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#######################
# PCA from scratch -- HW 2 part 1
#######################
# load data
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# plot some faces
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()


# part a (this is acceptable)
# standardize data
scaler = StandardScaler()
X = scaler.fit_transform(faces.data)
y = faces.target
print(X.shape)

# normalize data (this is also acceptable)
scaler = Normalizer()
X = scaler.fit_transform(faces.data)
y = faces.target
print(X.shape)

# part b
# caclulate the mean of the data and remove the mean from the data
mean = np.mean(X, axis=0)
X = X - mean
print(X.shape)

# principal components analysis (PCA) for dimensionality reduction
# the previous steps are wrapped into a function for convenience
def pca(X, k):
    # X is a 2D array of size n x d
    # k is the number of principal components
    # returns a 2D array of size n x k
    # normalize the data
    X = X - X.mean(axis=0)
    # compute the covariance matrix
    R = np.cov(X, rowvar=False)

    # compute eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eigh(R)

    # sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]

    # sort eigenvectors according to same index
    evals = evals[idx]

    # select the first k eigenvectors (projected data)
    evecs = evecs[:, :k]

    return np.dot(evecs.T, X.T).T


# part c
# compute the covariance matrix of your data
R = np.cov(X, rowvar=False)
print(R.shape)

# part d
# compute the eigenvalues and eigenvectors of the covariance matrix
evals, evecs = np.linalg.eigh(R)
print(evals.shape)
print(evecs.shape)

# sort eigenvalues in decreasing order
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]

# sort eigenvectors according to same index
evals = evals[idx]

# part e
# select the first k eigenvectors (projected data)
k = 2
evecs = evecs[:, :k]

# part f
# project the data onto the first k eigenvectors
pca_X = np.dot(evecs.T, X.T).T
print(pca_X.shape)


#######################
# PCA from scratch -- HW 2 part 2
#######################
# inverse pca
# PCA reconstruction=PC scores⋅Eigenvectors (transpose) +Mean
def inv_pca(pca_X, evecs, mean):
    # pca_X is a 2D array of size n x k
    # evecs is the eigenvectors of the covariance matrix
    # mean is the mean of the original data
    # returns a 2D array of size n x d
    return np.dot(pca_X, evecs.T) + mean

X_approx = inv_pca(pca_X, evecs, mean)

# compute the pariwise distance matrix of the original faces data
# and the reconstructed faces data
# use the euclidean distance
# plot the distance matrix as a heatmap

# pairwise distance matrix of PCA of k (2 in this example) components
X_pca = pca(X, 2)
dist_pca = np.zeros((len(X_pca), len(X_pca)))
for i in range(len(X_pca)):
    for j in range(len(X_pca)):
        dist_pca[i,j] = np.linalg.norm(X_pca[i] - X_pca[j])

# plot the distance matrix as a heatmap
sns.heatmap(dist_pca)
plt.show()

# inverse pca
dist_xapprox = np.zeros((len(X_approx), len(X_approx)))
for i in range(len(X_approx)):
    for j in range(len(X_approx)):
        dist_xapprox[i,j] = np.linalg.norm(X_approx[i] - X_approx[j])

# plot the distance matrix as a heatmap
sns.heatmap(dist_xapprox)

# compute the average difference between the two distance matrices
diff = np.abs(dist_pca - dist_xapprox)
avg_diff = np.mean(diff)


#######################
# PCA from scratch -- HW 2 part 3
#######################

#######################
# part a
# The ideal fit would be a ling a line with a slope of 1 and an intercept of 0.

# part b
X = faces.data
dist = np.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        dist[i,j] = np.linalg.norm(X[i] - X[j])

# plot the distance matrix as a heatmap
sns.heatmap(dist)
plt.show()

# compute the average difference between the two distance matrices
diff2 = np.abs(dist - dist_xapprox)
avg_diff = np.mean(diff2)
print (avg_diff)

# part c average error compared to ideal fit
# it's the same as the average difference between the two distance matrices

# part d

# flatten each distance matrix and plot them to campare
dist_flat = dist.flatten()
dist_pca_flat = dist_pca.flatten()
dist_xapprox_flat = dist_xapprox.flatten()
plt.scatter(dist_flat, dist_pca_flat, label='PCA vs. original')
# plt.scatter(dist_flat, dist_xapprox_flat, label='reconstructed vs. original')
# plt.legend()
plt.show()


#######################
# PCA from scratch -- HW 2 part 4
#######################
# project the faces to 2d using mds
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed')
X_mds = mds.fit_transform(dist)

# compute the distnace matrix of the mds projected data
dist_mds = np.zeros((len(X_mds), len(X_mds)))
for i in range(len(X_mds)):
    for j in range(len(X_mds)):
        dist_mds[i,j] = np.linalg.norm(X_mds[i] - X_mds[j])

# plot the distance matrix as a heatmap
sns.heatmap(dist_mds)
plt.show()

# part a
# the ideal fit would be a line with a slope of 1 and an intercept of 0

# part b
# compute the average difference between the two distance matrices
diff3 = np.abs(dist_mds - dist)
avg_diff = np.mean(diff3)

# part c
# same as part b

# part d
# flatten and sort the original distance matrix and plot it as a scatter plot
dist_flat = dist.flatten()
dist_mds_flat = dist_mds.flatten()
plt.scatter(dist_flat, dist_mds_flat, label='MDS vs. original')
plt.legend()
plt.show()


