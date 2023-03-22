import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

class KMeans():

    def __init__(self,k = 3,max_iters = 100):  # No need to implement
        self.k = k
        self.max_iters = max_iters

    def pairwise_dist(self, x, y):  # [5 pts]

        xSumSquare = np.sum(np.square(x),axis=1);
        ySumSquare = np.sum(np.square(y),axis=1);
        mul = np.dot(x, y.T);
        dists = np.sqrt(abs(xSumSquare[:, np.newaxis] + ySumSquare-2*mul))
        return dists

    def _init_centers(self, points):  # [5 pts]

        row, col = points.shape
        retArr = np.empty([self.k, col])
        for number in range(self.k):
            randIndex = np.random.randint(row)
            retArr[number] = points[randIndex]

        return retArr

    def _update_assignment(self, centers, points):  # [10 pts]

        row, col = points.shape
        cluster_idx = np.empty([row])
        distances = self.pairwise_dist(points, centers)
        cluster_idx = np.argmin(distances, axis=1)

        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]

        K, D = old_centers.shape
        new_centers = np.empty(old_centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(points[cluster_idx == i], axis = 0)
        return new_centers

    def _get_loss(self, centers, cluster_idx, points):
        dists = self.pairwise_dist(points, centers)
        loss = 0.0
        N, D = points.shape
        for i in range(N):
            loss = loss + np.square(dists[i][cluster_idx[i]])
        return loss


    def call(self, points):

        iteration = 0
        centers = self._init_centers(points)
        prev_centers = None
        while np.not_equal(centers, prev_centers).any() and iteration < self.max_iters:
            cluster_idx = self._update_assignment(centers, points)
            prev_centers = centers
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            if np.isnan(centers).any():
                centers = prev_centers
            iteration += 1
            print('Running iteration: {}, loss: {}'.format(iteration,loss))
        return cluster_idx, centers, loss


running3000 = running.iloc[100:3100,].reset_index(drop=True).to_numpy()
walking3000 = walking.iloc[100:3100,].reset_index(drop=True).to_numpy()
jumping3000 = jumping.iloc[100:3100,].reset_index(drop=True).to_numpy()

combined_data = np.concatenate([running3000,walking3000,jumping3000])
model = KMeans(k = 3,max_iters = 100)
idx, c, loss = model.call(combined_data)

mds = MDS(n_components=200)
mds_data = mds.fit_transform(face_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(combined_data)

#customcmap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])

fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(x=pca_data[:,0], y=pca_data[:,1], #s=150,
            c=idx)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


def plot_elbow(data, max_K=10):
        y_val = np.empty(max_K)
        for i in range(max_K):
            model = KMeans(k = i+1,max_iters = 100)
            idx, c, y_val[i] = model.call(data)
        plt.plot(np.arange(max_K) + 1, y_val)
        plt.show()
        return y_val

plot_elbow(combined_data)

