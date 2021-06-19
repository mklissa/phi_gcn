from __future__ import division
from __future__ import print_function

import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import pdb
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import os
# from sklearn.cluster import SpectralClustering
# from sklearn import metrics

# import gcn.globs as g
from gcn.graphutils import *
from gcn.models import GCN, MLP
# from sim import *
tf.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS

lastoutputs= None

def get_full_graph(sess, seed, edges, vertices, adj, labels, source, sink, reward_states, other_sources=[],other_sinks=[]):


    
    features = np.eye(len(vertices), dtype=np.int32) # One-hot encoding for the features (i.e. feature-less)
    features = sparse_to_tuple(sp.lil_matrix(features))

    np.random.seed(seed)
    tf.set_random_seed(seed)


    y_train, y_val, train_mask, val_mask = get_splits(labels, source, sink, reward_states)


    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN

    adj =adj.toarray()
    deg = np.diag(np.sum(adj,axis=1))
    laplacian = deg - adj



    # Define placeholders
    placeholders = {
        'adj': tf.placeholder(tf.float32, shape=(None, None)) , #unnormalized adjancy matrix
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32)
    }





    model = model_func(placeholders,edges,laplacian, input_dim=features[2][1], logging=True, FLAGS=FLAGS)
    
    sess.run(tf.global_variables_initializer())


    feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})

    cost_val = []

    start = time.time()
    for epoch in range(FLAGS.epochs):




        t = time.time()
        feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})
        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.learning_rate], feed_dict=feed_dict)



    print("Total time for gcn {}".format(time.time()-start))
    print("Optimization Finished!")


    outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]


    return outputs[:,1]
