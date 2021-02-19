from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp


def update_graph(model, optimizer, features, adj, rew_states, loss, args,envs):
    if adj.shape[0] >1:
        labels = torch.zeros((len(features)))
        idx_train = torch.LongTensor([0])
        for r_s in rew_states:
            if len(envs.observation_space.shape) == 1 : #MuJoCo experiments
                labels[r_s[0]] = torch.sigmoid(2*r_s[1])
            else:
                labels[r_s[0]] = torch.tensor([1.]) if r_s[1] > 0. else torch.tensor([0.])
            idx_train=torch.cat((idx_train, torch.LongTensor([r_s[0]]) ), 0)
        labels= labels.type(torch.LongTensor)
    else:
        labels = torch.zeros((len(features))).type(torch.LongTensor)
        idx_train = torch.LongTensor([0])
   
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    if args.cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        laplacian =laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out= torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1],1)
        loss_reg  = torch.mm(torch.mm(soft_out.T,laplacian),soft_out)
        loss_train +=  args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()

        
