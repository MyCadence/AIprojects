from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

points = data[0]
labels = data[1]
feature_1 = points[:, 0]
feature_2 = points[:, 1]



# import the library
from sklearn.cluster import KMeans
# Create the KMeans object and specify its characteristics

plt.figure(figsize=(5, 25))#Resize sub plots for Sample Report
for i,k in enumerate(range(2, 7)):

    #Left Column (True Clusters)
    plt.subplot(5, 2, 2*i + 1)
    plt.scatter(feature_1, feature_2, c=labels, s=10, cmap='viridis')
    plt.title('True Clusters')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

    #Right Column (Predicted)
    kmeans_method = KMeans(n_clusters=k)
    kmeans_method.fit(points)
    labels_predicted = kmeans_method.predict(points)

    plt.subplot(5, 2, 2*i + 2)
    plt.scatter(feature_1, feature_2, c=labels_predicted, s=10, cmap='viridis')
    plt.title(f'Predicted Clusters: K={k}')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)

plt.subplots_adjust(hspace=0.6, wspace=0.2)
plt.show()

