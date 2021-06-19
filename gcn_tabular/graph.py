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
from sklearn.cluster import SpectralClustering
from sklearn import metrics

import gcn.globs as g
from gcn.graphutils import *
from gcn.models import GCN, MLP
from sim import *
# from transfer import *
import matplotlib.cm as cm
colors = [(0,0,0)] +[(0.5,0.5,0.5)]+ [(cm.viridis(i)) for i in range(2,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

flags = tf.app.flags
FLAGS = flags.FLAGS

lastoutputs= None

def get_graph(sess, seed, edges, vertices, adj, labels, source, sink, reward_states, env, other_sources=[],other_sinks=[]):

# def get_graph(seed,sess,reward_states,env,itera):

    row,col = env.grid.shape

    eachepoch = FLAGS.eachepoch # Plot the resulting VF after each epoch or not


    features = np.eye(len(vertices), dtype=np.int32) # One-hot encoding for the features (i.e. feature-less)
    features = sparse_to_tuple(sp.lil_matrix(features))

    np.random.seed(seed)
    tf.set_random_seed(seed)



    # features, adj, labels,\
    #  vertices, edges, row, col, source,\
    #   sink, other_sinks, obs2grid, grid2obs = load_data(append=FLAGS.app)

    # reward_states =  map(grid2obs.get,reward_states)

    # pdb.set_trace()
    y_train, y_val,\
     train_mask, val_mask = get_splits(labels, source, sink, reward_states)


    if sink is None:
        print("Sink not there, skipping this seed.")
        return None,None



    # Compute some baselines
    # start = time.time()
    # sc = SpectralClustering(2, affinity='precomputed', 
    #                         n_init=5, eigen_solver='arpack', assign_labels='discretize')
    # sc.fit(adj.toarray())
    # print('Time for spectral clustering {}'.format(time.time()-start))

    # start = time.time()
    # cut = nx.minimum_cut(G,source,sink)
    # print("Total time for mincut {}".format(time.time()-start))



    # Some preprocessing
    # features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    # adj=normalize_adj(adj).toarray()
    # pdb.set_trace()
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


        if eachepoch:
            outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]

            X = np.ones((row*col)) * -0.11
            Xround = np.ones((row*col)) *.5
            X[vertices] = outputs[:,1]
            walls = np.where(env.grid.flatten()==1)
            X[walls]=-0.1
            Xround[vertices] = np.round(X[vertices])
            if (Xround[vertices] == 0.).all():
                X[0] = Xround[0] = 1.
            X=X.reshape(row,col)
            Xround = Xround.reshape(row,col)



            # path = np.ones((row*col))*0.25
            # path[vertices] = .5
            # path_sources = map(obs2grid.get, other_sources)
            # path[path_sources] = 0.
            # path_sinks = map(obs2grid.get, other_sinks)
            # path[path_sinks] = 1.
            # path=path.reshape(row,col)


            # A=map(obs2grid.get,cut[1][0])
            # B=map(obs2grid.get,cut[1][1])
            # view_cut=np.ones((row*col))*0.5
            # view_cut[A] = 0
            # view_cut[B] = 1
            # view_cut=view_cut.reshape(row,col)


            # pdb.set_trace()

            # A=map(obs2grid.get,np.argwhere(sc.labels_ ==1).flatten() ) 
            # B=map(obs2grid.get,np.argwhere(sc.labels_ ==0).flatten() )

            # n_cut=np.ones((row*col))*0.5
            # n_cut[A] = 0
            # n_cut[B] = 1
            # n_cut=n_cut.reshape(row,col)
            # pdb.set_trace()

            # if featplot is not None:
            #     fig, ax = plt.subplots(6,1)
            #     ax[0].imshow(path, interpolation='nearest')
            #     ax[1].imshow(featplot.reshape(row,col), interpolation='nearest')
            #     ax[2].imshow(Xround, interpolation='nearest')
            #     ax[3].imshow(X, interpolation='nearest')
            #     ax[4].imshow(view_cut, interpolation='nearest')
            #     ax[5].imshow(n_cut, interpolation='nearest')
                
            # else:
            mask_map = np.ones((row*col)) * -0.11
            mask_map[vertices] = y_train[:,1]
            mask_map=mask_map.reshape(row,col)

            # fig, ax = plt.subplots(5,1)
            # ax[0].imshow(path, interpolation='nearest')
            # ax[1].imshow(Xround, interpolation='nearest')
            # ax[2].imshow(X, interpolation='nearest')
            # ax[3].imshow(view_cut, interpolation='nearest')
            # ax[4].imshow(n_cut, interpolation='nearest')
            fig, ax = plt.subplots(2,1)
            # ax[0].imshow(path, interpolation='nearest')
            ax[0].imshow(X, interpolation='nearest')   
            ax[1].imshow(mask_map, interpolation='nearest',cmap=new_map)                             



            # plt.show()
            # plt.close()


            directory = "diff_{}_{}/".format(FLAGS.app,row*col)
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig("{}seed{}_{}.png".format(directory,seed,epoch))
            plt.close();plt.clf()





        t = time.time()
        # pdb.set_trace()
        feed_dict = construct_feed_dict(adj, features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['learning_rate']: FLAGS.learning_rate})
        outs = sess.run([model.opt_op, model.loss, model.accuracy,model.learning_rate], feed_dict=feed_dict)



    print("Total time for gcn {}".format(time.time()-start))
    print("Optimization Finished!")






    if not eachepoch:

        outputs = sess.run([tf.nn.softmax(model.outputs)], feed_dict=feed_dict)[0]



        X = np.ones((row*col)) * -0.11
        Xround = np.ones((row*col)) *.5
        X[vertices] = outputs[:,1]
        walls = np.where(env.grid.flatten()==1)
        X[walls]=-0.1
        Xround[vertices] = np.round(X[vertices])
        if (Xround[vertices] == 0.).all():
            X[0] = Xround[0] = 1.
        X=X.reshape(row,col)
        Xround = Xround.reshape(row,col)


        mask_map = np.ones((row*col)) * -0.11
        mask_map[vertices] = y_train[:,1]
        mask_map=mask_map.reshape(row,col)


        # fig, ax = plt.subplots(1)      
        # ax.imshow(X, interpolation='nearest',cmap=new_map)
        # # ax[1].imshow(mask_map, interpolation='nearest',cmap=new_map)
        # # plt.show()
        # # plt.close()
        # # sys.exit()


        # directory = "{}_{}/".format(FLAGS.app,row*col)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # if FLAGS.fig:
        #     myfig = '_'+FLAGS.fig + "_{}".format(epoch)
        # else:
        #     myfig = ''

        # # plt.title('Diffusion-Based Approximate Value Function')
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig("{}seed{}_{}.png".format(directory,seed,myfig),bbox_inches='tight')
        # plt.close()
        # # pdb.set_trace()
        # sys.exit()



    # pdb.set_trace()
    initiation_set = list(np.argwhere(Xround.flatten()[vertices] == 0))
    goals = np.argwhere(Xround.flatten()[vertices] == 1)



    # if len(initiation_set) > 1:
    #     initiation_set = map(obs2grid.get,initiation_set.squeeze())
    # if len(goals) > 1:
    #     goals = map(obs2grid.get,goals.squeeze())
    V_weights = outputs[:,1]

    # pdb.set_trace()

    # sinks = list(np.argsort(X.flatten())[::-1])
    # sinks = [sink for sink in sinks if sink in goals]

    # pdb.set_trace()

    tf.reset_default_graph()
    return initiation_set, V_weights

