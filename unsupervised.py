# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
# import seaborn as sns
# import matplotlib.pyplot as plt 
# X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# # sns.scatterplot(x=X[:, 0], y=X[:, 1])
# # plt.show()

# kmeans = KMeans(n_clusters=20)
# y_kmeans = kmeans.fit_predict(X) # [0,0,0,1,1,1]
# # print(y_kmeans)
# centers = kmeans.cluster_centers_
# print(centers)

# # Nubraižome taškus, nuspalvintus pagal klasterius
# sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='viridis', s=50, alpha=0.7)

# # Pažymime klasterių centrus (Centroidus)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroidai')

# plt.title("K-means klasterizacija 2D erdvėje")
# plt.legend()
# plt.show()


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
X, y_true = make_blobs(n_samples=300, centers=50, cluster_std=0.60, random_state=42)

sns.scatterplot(x=X[:, 0], y=X[:, 1])
plt.show()
k_values = range(1,10)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    inertia_values.append(kmeans.inertia_) 
# print(y_kmeans)
# centers = kmeans.cluster_centers_
# print(centers)

sns.lineplot(x=k_values,y=inertia_values)
plt.show()

kmeans = KMeans(n_clusters=7)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
print(centers)

# Nubraižome taškus, nuspalvintus pagal klasterius
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='viridis', s=50, alpha=0.7)


# Pažymime klasterių centrus (Centroidus)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroidai')

plt.title("K-means klasterizacija 2D erdvėje")
plt.legend()
plt.show()