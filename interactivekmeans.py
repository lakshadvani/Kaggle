from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
np.random.seed(42)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpldatacursor import datacursor
df = pd.read_csv('all.csv')
#arr = df.values
X = df.drop('iscrypto',1)

X = df.drop('Name',1)
y = df['iscrypto'] 
names = df['Name']
X= X.values
#print(X)
data = X

n_samples, n_features = data.shape
n_targets = 2
labels = y.values
'''
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_targets, n_samples, n_features))


print(82 * '_')
'''

nnn = int(input("N of custers "))

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1



# #############################################################################
# Visualize the results on PCA-reduced data


reduced_data = PCA(n_components=2).fit_transform(data)




h = .02

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))




kmeans = KMeans(init='k-means++', n_clusters=nnn, n_init=10)
Z = kmeans.fit(reduced_data)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
############################


############
fig,ax = plt.subplots()

dd = plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')


for i in range(len(reduced_data)):
	if int(labels[i]) == 1:
		plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k.', markersize=12,label = names[i])
	else:
		plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'g^', markersize=12,label = names[i])

legend_elements = [Line2D([0], [0], marker='^', color='g', label='Non Cryptographic',markerfacecolor='g', markersize=12), Line2D([0], [0], marker='.', color='black', label='Cryptographic',markerfacecolor='black', markersize=12)]

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
plt.title('K-means clustering on the history.csv dataset (PCA-reduced data) using:'+str(nnn)+' clusters' )
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.ylabel('Second Principal Component')
plt.xlabel('First Principal Component')
plt.legend(handles=legend_elements)

datacursor(formatter='{label}'.format)

plt.show()
'''
for i in range(len(X)):
	print(names[i],labels[i],kmeans.labels_[i])
'''
#############

'''
for i in range(len(reduced_data)):
    plt.plot(reduced_data[i, 0], reduced_data[i, 1], 'k.', markersize=5,label = names[i])
datacursor(formatter='{label}'.format)

line, = plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)
'''
