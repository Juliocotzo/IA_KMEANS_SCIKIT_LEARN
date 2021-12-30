import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data = [[8,2],[9,7],[2,12],[9,1],[10,7],[3,14],[8,1],[1,13]]
x=np.array(data)

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

print(kmeans.cluster_centers_)
plt.scatter(x[:,0],x[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title("Kmeans - Scikit Learn" ,fontsize=10)
plt.xlabel('Efectividad')
plt.ylabel('pH')
plt.show()