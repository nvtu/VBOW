from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter, defaultdict
from sklearn.externals import joblib


X_feat = np.load('combined_sift_feat.npy')
kmeans = MiniBatchKMeans(n_clusters = 512, random_state=0, batch_size = 1000, max_iter=500).fit(X_feat)
joblib.dump(kmeans, 'kmeans_cluster.storage')
#print(kmeans.cluster_centers_)
# print(Counter(kmeans.labels_))