import numpy as np, pickle as pkl, networkx as nx, scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, matplotlib.pyplot as plt, pdb

def load_other(path='gcn/data/', append='', force_feats=False, force_nofeats=False):
    
    info = np.loadtxt(('{}{}_info.txt').format(path, append))
    row, col = map(int, (info[(0, 0)], info[(0, 1)]))
    source, sink = map(int, (info[(1, 0)], info[(1, 1)]))
    size = row * col
    vertices, features = map(int, info[2:, 0]), info[2:, 1]
    edges = np.loadtxt(('{}{}_edges.txt').format(path, append), dtype=np.int32)
    graph_dict = dict(zip(vertices, range(len(vertices))))
    featplot = None
    nofeats = (features == -1).sum()
    if nofeats or force_nofeats:
        features = np.eye(len(vertices), dtype=np.float32)
    else:
        features = features[(Ellipsis, None)]
        features = np.hstack((1 - features, features))
        featplot = np.ones(row * col) * -0.1
        featplot[vertices] = features[:, 1]
        featplot = featplot.reshape(row, col)
    if force_feats:
        features = np.hstack((np.random.uniform(0.5, 0.8, size=(size, 1)),
         np.random.uniform(0.2, 0.5, size=(size, 1))))
        for r in range(row):
            features[row * r + 5:row * (r + 1), :] = np.roll(features[row * r + 5:row * (r + 1), :], 1, axis=1)

        featplot = features[:, 1]
        features = preprocess_features(features)
        features = features[vertices]
    other_sinks = None
    source, sink = get_source_sink(source, sink, features, vertices, graph_dict)
    labels = np.zeros(len(vertices))
    labels[sink] = 1
    labels = encode_onehot(labels)
    features = sparse_to_tuple(sp.lil_matrix(features))
    graphsize = reduce(lambda x, y: x * y, edges.shape)
    edges = np.array(map(graph_dict.get, edges.reshape(graphsize))).reshape(edges.shape)
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    graph_dict = {v:k for k, v in graph_dict.iteritems()}
    return (
     features, featplot, adj, labels, vertices, edges, row, col, source, sink, other_sinks, graph_dict)


def get_source_sink(source, sink, feats, vertices, graph_dict):
    if source == -1 or sink == -1:
        if feats.shape == np.eye(feats.shape[0]).shape and (feats == np.eye(feats.shape[0])).all():
            choices = np.random.choice(len(vertices), 2, replace=False)
            source = choices[0]
            sink = choices[1]
        else:
            source = np.argmin(feats[:, 1])
            sink = np.argmax(feats[:, 1])
    else:
        source = graph_dict.get(source)
        sink = graph_dict.get(sink)
    return (
     source, sink)


def get_splits(y, source, sink, other_sinks):
    if other_sinks is not None:
        idx_train = [
         source, sink, other_sinks[-1][0]]
        # print idx_train
    else:
        idx_train = [
         source, sink]
    idx_val = range(len(y))
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    return (
     y_train, y_val, train_mask, val_mask)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_normalized = adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj, features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['support'][i]:support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print ('Calculating Chebyshev polynomials up to order {}...').format(k)
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = 2.0 / largest_eigval[0] * laplacian - sp.eye(adj.shape[0])
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return (
         coords, values, shape)

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])

    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



    



def make_interpolater(left_min, left_max, right_min, right_max): 
    if left_min == left_max:
        return lambda x:x
    # Figure out how 'wide' each range is  
    leftSpan = left_max - left_min  
    rightSpan = right_max - right_min  

    # Compute the scale factor between left and right values 
    scaleFactor = float(rightSpan) / float(leftSpan) 

    # create interpolation function using pre-calculated scaleFactor
    def interp_fn(value):
        return right_min + (value-left_min)*scaleFactor

    return interp_fn
        