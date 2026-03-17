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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, fowlkes_mallows_score
X, y_true = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=42)

sns.scatterplot(x=X[:, 0], y=X[:, 1])
plt.show()
# k_values = range(1,10)
# inertia_values = []

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     y_kmeans = kmeans.fit_predict(X)
#     inertia_values.append(kmeans.inertia_) 
# print(y_kmeans)
# centers = kmeans.cluster_centers_
# print(centers)

# sns.lineplot(x=k_values,y=inertia_values)
# plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
np.random.seed(42)

Z = linkage(X)
print(Z.shape)
# dendrogram(Z)
# plt.show()


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, dendrogram

# 1. Nustatome atstumą (t), ties kuriuo nupjausime dendrogramą
t = 4  # Pakeisk šį skaičių pagal savo dendrogramos y ašį
clusters = fcluster(Z, t, criterion='distance')

# 2. Atvaizduojame dendrogramą su horizontalia pjūvio linija
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.axhline(y=t, c='red', lw=2, linestyle='--') # Raudona brūkšninė linija ties t
plt.title(f'Dendrograma su pjūviu ties atstumu {t}')
plt.xlabel('Duomenų taškai')
plt.ylabel('Atstumas')
plt.show()

# 3. Pridedame klasterius prie X ir atvaizduojame sklaidos diagramą (Scatter plot)
# Kadangi X turi 2 stulpelius:
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism', s=50)
plt.title(f'Klasteriai (nupjauta ties atstumu {t})')
plt.colorbar(label='Klasterio ID')
plt.show()

# Jei nori pamatyti, kiek klasterių gavosi:
n_clusters = len(set(clusters))
print(f"Gauta klasterių: {n_clusters}")

# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import fcluster

# # 1. Nustatome, į kiek klasterių norime skirstyti (pvz., k = 3)
# # Arba galime nustatyti atstumą (t), bet 'maxclust' yra paprastesnis būdas
# k = 6 
# clusters = fcluster(Z, 4)

# # 2. Pridedame klasterių etiketes prie X
# # Jei X yra numpy masyvas:
# X_with_clusters = np.column_stack((X, clusters))

# # 3. Atvaizduojame rezultatus (kadangi turi 2 stulpelius)
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
# plt.title(f"Duomenų suskirstymas į {k} klasterius")
# plt.xlabel("Požymis 1")
# plt.ylabel("Požymis 2")
# plt.colorbar(label='Klasterio Nr.')
# plt.show()

# kmeans = KMeans(n_clusters=4)
# y_kmeans = kmeans.fit_predict(X)
# centers = kmeans.cluster_centers_
# print(centers)
# print(calinski_harabasz_score(X,y_kmeans))
# print(davies_bouldin_score(X,y_kmeans))
# print(silhouette_score(X, y_kmeans))
# # Nubraižome taškus, nuspalvintus pagal klasterius
# sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette='viridis', s=50, alpha=0.7)


# # Pažymime klasterių centrus (Centroidus)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroidai')

# plt.title("K-means klasterizacija 2D erdvėje")
# plt.legend()
# plt.show()