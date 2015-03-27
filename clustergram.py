import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import json
from scipy.stats import zscore
from collections import Counter
import networkx as nx
from networkx.readwrite import json_graph

def _linkageMatrix2json(Z, labels):
	## function to convert linkage matrix to a json dict through DiGraph object
	n = Z.shape[0] + 1
	G = nx.DiGraph()
	for i, row in enumerate(Z):
		cluster = int(n + i)
		child0 = int(row[0])
		if child0 < n:
			child0 = labels[child0]
			G.add_node(child0, height=0)
		child1 = int(row[1])
		if child1 < n:
			child1 = labels[child1]
			G.add_node(child1, height=0)
		G.add_edge(cluster, child0)
		G.add_edge(cluster, child1)
		G.node[cluster]['height'] = row[2]
	root = cluster
	json_data = json_graph.tree_data(G,root=root)
	return json_data


def clustergram(data=None, row_labels=None, col_labels=None,
			row_linkage='average', col_linkage='average', 
			row_pdist='euclidean', col_pdist='euclidean',
			standardize=3, log=False, prefix=None):
	"""
	A function to output json of a clustergram given input data

	Parameters:
	----------
	data: a numpy array or numpy matrix
	row_labels: a list of strings corresponding to the rows in data
	col_labels: a list of strings corresponding to the columns in data
	cluster: boolean variable specifying whether to perform hierarchical clustering (True) or not (False)
	row_linkage: linkage method used for rows
	col_linkage: linkage method used for columns 
		options = ['average','single','complete','weighted','centroid','median','ward']
	row_pdist: pdist metric used for rows
	col_pdist: pdist metric used for columns
		options = ['euclidean','minkowski','cityblock','seuclidean','sqeuclidean',
		'cosine','correlation','hamming','jaccard','chebyshev','canberra','braycurtis',
		'mahalanobis','wminkowski']
	standardize: specifying the dimension for standardizing the values in data
		options =  {1: 'standardize along the columns of data',
					2: 'standardize along the rows of data',
					3: 'do not standardize the data'}
	log: boolean variable specifying whether to perform log transform for the data

	Example:
	----------
	from clustergram import clustergram
	clustergram(data=np.random.randn(3,3),row_labels=['a','b','c'],
		col_labels=['1','2','3'], row_groups=['A','A','B'],
		col_groups=['group1','group1','group2'])

	"""
	## preprocess data
	if log:
		data = np.log2(data + 1.0)

	if standardize == 1: # Standardize along the columns of data
		data = zscore(data, axis=0)
	elif standardize == 2: # Standardize along the rows of data
		data = zscore(data, axis=1)

	## perform hierarchical clustering for rows and cols
	## compute pdist for rows:
	d1 = dist.pdist(data, metric=row_pdist)
	D1 = dist.squareform(d1)
	Y1 = sch.linkage(D1, method=row_linkage, metric=row_pdist)
	Z1 = sch.dendrogram(Y1, orientation='right')
	idx1 = Z1['leaves']
	json_data1 = _linkageMatrix2json(Y1, row_labels)

	## compute pdist for cols
	d2 = dist.pdist(data.T, metric=col_pdist)
	D2 = dist.squareform(d2)
	Y2 = sch.linkage(D2, method=col_linkage, metric=col_pdist)
	Z2 = sch.dendrogram(Y2)
	idx2 = Z2['leaves']
	json_data2 = _linkageMatrix2json(Y2, col_labels)

	## plot heatmap
	data_clustered = data
	data_clustered = data_clustered[:,idx2]
	data_clustered = data_clustered[idx1,:]
	if prefix is None:
		return data_clustered, json_data1, json_data2
	else:
		json.dump(data_clustered.tolist(), open(prefix + '_matrix.json', 'wb'))
		json.dump(json_data1, open(prefix + '_dendroRow.json', 'wb'))
		json.dump(json_data2, open(prefix + '_dendroCol.json', 'wb'))
		return


# data, d1, d2 = clustergram(data=np.random.randn(4,3), row_labels=['a','b','c','d'],col_labels=['1','2','3'])

import os
os.chdir('json')
# clustergram(data=np.random.randn(6,3), row_labels=['a','b','c','d','e','f'],col_labels=['1','2','3'],prefix='testRand')
clustergram(data=np.random.randn(100,10), row_labels=map(str,range(100)),col_labels=map(str,range(10)),prefix='testRand')
