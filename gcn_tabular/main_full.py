import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
from utils import *
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os 
import networkx as nx
from graphfull import get_full_graph

import pdb
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import collections

colors = [(0,0,0)] +[(0.5,0.5,0.5)]+ [(cm.viridis(i)) for i in range(2,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


plotsteps=0



flags.DEFINE_integer('gcn', 1, 'Do you want to generate a graph?')
flags.DEFINE_integer('mp', 0, 'Do you want to generate a message passing baseline?')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', 8e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('nf', 1, 'Create features or not.')
flags.DEFINE_integer('f', 0, 'Create features or not.')
flags.DEFINE_string('fig', '', 'Figure identifier.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('app', '', 'For data file loading') 
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('eachepoch', 0, 'Plot difusion at each epoch or not.')
flags.DEFINE_integer('path', 0, 'Plot each epochs path.')
flags.DEFINE_integer('plots', 0, 'Plot the value function')



 
flags.DEFINE_integer('ngraph', 10, "Number of episodes before graph generation")
flags.DEFINE_integer('nepisodes', 300, "Number of episodes per run")
flags.DEFINE_integer('nruns',1, "Number of runs")
flags.DEFINE_integer('nsteps',1000, "Maximum number of steps per episode")
flags.DEFINE_integer('noptions',1, 'Number of options')
flags.DEFINE_integer('baseline',1, "Use the baseline for the intra-option gradient")
flags.DEFINE_integer('primitive',0, "Augment with primitive")



flags.DEFINE_float('temperature',1e-1, "Temperature parameter for softmax")
flags.DEFINE_float('discount',0.99, 'Discount factor')
flags.DEFINE_float('lr_intra',1e-1, "Intra-option gradient learning rate")
flags.DEFINE_float('lr_critic',1e-1, "Learning rate")




flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 8, 'Number of units in hidden layer 1.')



want_graph= FLAGS.gcn
seeds = [FLAGS.seed]
totalsteps = []

for seed in seeds:
    print('seed:',seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rng = np.random.RandomState(seed)

    
    env = Fourrooms()
    walls = np.argwhere(env.grid.flatten()==1)
    row,col = env.grid.shape




    ################ Message passing ################
    

    acts = [-col,col,-1,1]
    allG = nx.Graph()
    for s in env.possible_states[:,0]:
        for a in acts:
            if (s + a) in env.possible_states[:,0]:
                allG.add_edge(env.grid2obs.get(s), env.grid2obs.get(s+a)) 

    def breadth_first_search(graph, goal, start, more_rew, phis): 
        visited, queue = set([goal]), collections.deque([goal])
        while queue: 
            vertex = queue.popleft()
            neighbors = list(graph.neighbors(vertex))
            if vertex==goal:
                phis[vertex] = 1.
            elif vertex==start:
                phis[vertex] = 0.5
            elif vertex in more_rew:
                phis[vertex] = 0.
            else:
                n_vals = [phis[n] for n in neighbors]
                phis[vertex] = np.mean(n_vals) * 1.

            for neighbor in neighbors: 
                if neighbor not in visited: 
                    visited.add(neighbor) 
                    queue.append(neighbor) 
        return phis


    more_rew =env.more_rewards
    gridgoal=env.goal
    start=0
    mp = np.zeros((max(allG)+1,))
    for i in range(300):
        mp = breadth_first_search(allG, gridgoal, start, more_rew, mp)


    ################ Message passing ################



    

    ############ GCN ##############

    acts = [-col,col,-1,1]
    allG = nx.Graph()
    for s in env.possible_states[:,0]:
        for a in acts:
            if (s + a) in env.possible_states[:,0]:
                allG.add_edge(env.grid2obs.get(s), env.grid2obs.get(s+a)) 


    
    allobs = np.array(allG.nodes())
    edges = np.array(allG.edges())



    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(allobs), len(allobs)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # build symmetric adjacency matrix

    
    source=0
    sink=env.goal
    labels=np.zeros((len(allobs),2))
    labels[source] = [0.5, 0.5] # start state has  reward of 0.
    labels[sink] = [0., 1.]  # goal has a reward of 1.
    mapped_reward_states=[]
    for r in env.more_rewards:
        labels[r] =  [1.,0.]
        mapped_reward_states.append(r)
    
    sess = tf.Session()

    gcn_dist = get_full_graph(sess, seed, edges, allobs, adj,
                                    labels, source, sink, mapped_reward_states)
    ############ GCN ##############




    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n


    observations = set()
    G = nx.Graph()
    G.add_nodes_from(range(len(env.grid.flatten())))


    options_grid2obs = [env.grid2obs, ]

    option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, FLAGS.temperature,FLAGS.lr_intra), ]

    critics = [StateValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures)) ), ]
    action_critics = [ActionValue(FLAGS.discount, FLAGS.lr_critic, np.zeros((nfeatures,nactions)) ), ]

    first=True
    done=False
    cumsteps = 0.
    optionsteps = 0.
    myrand = 1.
    myrandinit =1.
    sources=[]
    reward_states = set()

    init_set=[]
    goals = []
    allstates=[]
    for episode in range(FLAGS.nepisodes):
        epoch_states= set()
        rewards = []
        pos = env.grid.flatten().astype(float)
        pos[pos == 1] = -.5
        
        observation = env.reset()
        start=observation
        sources.append(start)
        observations.add(start)

        option=0



        last_phi = phi = options_grid2obs[option].get(observation)

        action = option_policies[option].sample(phi,0)  if np.random.rand() > 0.1 else np.random.randint(nactions)
        critics[option].save(phi)
        action_critics[option].save(phi, action)
        option_policies[option].save(phi,action)

        episode_states= []
        episode_states.append(phi)
        cumreward = 0.
        for step in range(FLAGS.nsteps):



            next_observation, reward, done, _ = env.step(action)
            real_reward = reward
            rewards.append(reward)
            if observation != next_observation:
                G.add_edge(observation,next_observation)

            if not first and want_graph:
                bonus = (0.99* (gcn_dist[env.grid2obs.get(next_observation)]) - (gcn_dist[env.grid2obs.get(observation)]) ) 
                reward += 0.5*bonus
            if not first and FLAGS.mp:
                bonus = (0.99* (mp[env.grid2obs.get(next_observation)]) - (mp[env.grid2obs.get(observation)]) ) 
                reward += 0.5*bonus

            if env.grid2obs.get(next_observation) == env.goal:
                first=False


            observation=next_observation
            observations.add(observation)
            epoch_states.add(observation)
            if reward and env.grid2obs.get(next_observation) != env.goal:
                reward_states.add(observation)
            

            pos[observation] += 0.1    
            if option ==1 or observation in map(env.obs2grid.get,env.init_states):
                optionsteps += 1


            #### Updates
            phi = options_grid2obs[option].get(observation)
            critics[option].update_lambda(phi, reward, done)
            action_critics[option].update(phi, reward, done, critics[option].value(phi))

            critic_feedback = reward + (1.-done) * FLAGS.discount * critics[option].value(phi)
            critic_feedback -= critics[option].last_value
            option_policies[option].update(critic_feedback)


            action = option_policies[option].sample(phi,step) if np.random.rand() > 0.1 else np.random.randint(nactions)

            critics[option].save(phi)
            action_critics[option].save(phi, action)
            option_policies[option].save(phi,action)

            if phi not in episode_states:
                episode_states.append(phi)
            cumreward += real_reward
            last_phi = phi
            if done:
                break



        cumsteps += step
        print('Episode {} steps {} cumreward {} cumsteps {} optionsteps {}'.format(episode, step, cumreward, cumsteps,optionsteps))


        critics[option].init_elig() # Reset eligibility trace



        totalsteps.append(cumsteps)

        import os
        if not os.path.exists('res'):
            os.makedirs('res')
        # Saving results
        savefile = "{}{}_graph{}_mp{}_seed{}.csv".format(row*col,FLAGS.app,FLAGS.gcn,FLAGS.mp,FLAGS.seed)
        np.savetxt("res/{}".format(savefile),totalsteps,delimiter=',')

