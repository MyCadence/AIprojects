from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

points = data[0]
labels = data[1]

feature_1 = points[:, 0]
feature_2 = points[:, 1]
plt.subplot(3, 2, 1)
plt.scatter(feature_1, feature_2, c=labels, s=10, cmap='viridis')
plt.title('True Clusters')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# plt.show()

# import the library
from sklearn.cluster import KMeans
# Create the KMeans object and specify its characteristics
for k in range(2, 7):
    kmeans_method = KMeans(n_clusters=k)
    kmeans_method.fit(points)

    # Predict the cluster for a set of points
    labels_predicted = kmeans_method.predict(points)

    plt.subplot(3, 2, k)
    plt.scatter(feature_1, feature_2, c=labels_predicted, s=10, cmap='viridis')
    plt.title(f'Predicted Clusters: K={k}')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()