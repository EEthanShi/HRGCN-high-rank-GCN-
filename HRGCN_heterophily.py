#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:09:19 2022

@author: daishi
"""

import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
#from GraphRicciCurvature.Xfeature import OllivierRicci
#from GraphRicciCurvature.My_OllivierRicci import OllivierRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon 
from torch_geometric.utils import get_laplacian, to_networkx
import argparse
import os.path as osp
import time
import scipy
import scipy.sparse as sp
from scipy.spatial import distance
import torch.optim as optim
from layers import GraphConvolution
import torch_geometric.transforms as T
from torch.optim import Adam, lr_scheduler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)




def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask


#%% models 
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, scale):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.scale = scale 

    def forward(self, x, adj):
        #x = F.rrelu(self.gc1(x, adj),upper=self.scale,lower =self.scale) 
        x = F.leaky_relu(self.gc1(x, adj),negative_slope=self.scale)
        #x= F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x,dim=1)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
  
    parser.add_argument('--dataset', type=str, default='wisconsin', # for chameleon bigger dataset, we must assign bigger nhid i.e.32
                        help='name of dataset (default: Cora): Cora, Citeseer, Pubmed,Coauthor Physics Coauthor CS Amazon Computer Amazon Photo')
   
    parser.add_argument('--reps', type=int, default=12,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.05, 
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=1e-5, 
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--nhid', type=int, default=16,   
                        help='number of hidden units (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.95,  
                        help='dropout probability (default: 0.7)')
    parser.add_argument('--Ollivier_alpha', type = float, default = 0.4, 
                        help='Ollivier_alpha value in Ollivier-Ricci Curvature (default: 0.7)')

    parser.add_argument('--noiseLev', type=float, default=0,   
                        help='Added noise level (default: 0.05 for 5% noise)')          
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1000)')
    parser.add_argument('--ExpNum', type=int, default='1',
                        help='The Experiment Number (default: 1)')
    parser.add_argument('--iteration_number', type=int, default=300,
                        help='number of iterations in the ricci flow evolution, based on the paper is ranged from 20-50')
    parser.add_argument('--CurvatureType', type=str, default='Ollivier',
                        help='Ricci curvature type: Ollivier (default) or Forman or None')
    parser.add_argument('--num_nbr', type=int, default=12000,
                        help='k edge weight neighbors for density distribution,Smaller k run faster but the result is less accurate. (Default value = 3000)')
    parser.add_argument('--scale', type=float, default=0.1, help='upper and lower limit for the scaling value of the negative polyhedra reconstruction') 

    parser.add_argument('--gamma', type=float, default=1, help='gamma for the similiarty measure function')
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.6, 
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.2)
    args = parser.parse_args()
    print(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count())
#%%
start_time = time.time()
dataname = args.dataset
rootname = osp.join(osp.abspath(''), 'data', dataname)

########## load heterophily dataset 

dataset = WebKB(rootname, dataname)
#dataset = Actor(rootname)
#dataset = WikipediaNetwork(rootname, dataname)

num_features = dataset.num_features
num_classes = dataset.num_classes

data = dataset[0]
num_nodes = data.x.shape[0]
rand_seed = args.seed
num_train = int(len(data.y) / num_classes * args.train_rate)
num_val = int(len(data.y) / num_classes * args.val_rate)
data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, rand_seed, num_train, num_val)

features = data.x
features = normalize_features(features) ##################
G = to_networkx(data, to_undirected=False)
orc = OllivierRicci(G, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=args.num_nbr,exp_power=1) # should use 1 here since in the Olliviier compuation, we want to assign samilarity measure. 
orc.compute_ricci_curvature()  

G_orc = orc.G.copy()
rc = np.array(list(nx.get_edge_attributes(G_orc, 'ricciCurvature').values()))
adj = np.zeros((num_nodes, num_nodes))



for n1, n2 in G_orc.edges():   # re-define the ricci curature as k/dij to preserve the sign
    
  
    adj[n1, n2] = np.exp(-(orc.G[n1][n2]["ricciCurvature"])/(distance.euclidean(features[n1], features[n2])+np.random.uniform(0.1, 10**(-20))/10000))
    
    adj[n2, n1] = adj[n1, n2]
from numpy import inf
adj[adj == inf] = np.exp(100)

adj = sp.coo_matrix(adj) 
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#adj = normalize_adj(adj)
#adj = adj + sp.eye(adj.shape[0])
adj= torch.FloatTensor(np.array(adj.todense()))


features = normalize_features(features)
features = torch.FloatTensor(np.array(features))
labels = data.y
data = data.to(device)

#%%
'''training'''
if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    #adj_2=adj_2.cuda()
    labels = labels.cuda()
    
for i in range(10):

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))


    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    SaveResultFilename = args.dataset + 'Exp{0:03d}'.format(args.ExpNum)
    ResultCSV = args.dataset + 'RF_GCN_Lin.csv'
    
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)
    
    
    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        record_test_acc = 0.0
        model = GCN(nfeat=features.shape[1],
            nhid=args.nhid,
            nclass=labels.max().item() + 1,
            
            dropout=args.dropout,scale=0.5).to(device)
        optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)
        
        #0.5 ==》0.8259
        
        # training
        for epoch in range(num_epochs):
            # training mode            
            t = time.time()
            model.train()
            optimizer.zero_grad()
            #torch.set_default_tensor_type(torch.DoubleTensor)
            output = model(features, adj)
            #output = model(data,adj)
            loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])           
            loss_train.backward()
            optimizer.step()
            
        # evaluation mode
            model.eval()
            output = model(features, adj)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = output[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc
                e_loss = F.nll_loss(output[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss
            # print out results
            if (epoch + 1) % 2 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model   We dont need this on HPC
            if epoch > 10:
               if epoch_acc['val_mask'][rep, epoch] > max_acc:
                   #torch.save(model.state_dict(), SaveResultFilename + '.pth')
                   # print('Epoch: {:3d}'.format(epoch + 1),
                   #      'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   #      'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   #      'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   #      'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   #      'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   #      'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))
                   print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                   max_acc = epoch_acc['val_mask'][rep, epoch]
                   record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    # if osp.isfile(ResultCSV):
    #     df = pd.read_csv(ResultCSV)
    # else:
    #     outputs_names = {name: type(value).__name__ for (name, value) in args._get_kwargs()}
    #     outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1,num_reps+1)})
    #     outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
    #     df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    # new_row = {name: value for (name, value) in args._get_kwargs()}
    # new_row.update({'Replicate{0:2d}'.format(ii): saved_model_test_acc[ii-1] for ii in range(1,num_reps+1)})
    # new_row.update({'Ave_Test_Acc': np.mean(saved_model_test_acc), 'Test_Acc_std': np.std(saved_model_test_acc)})
    # df = df.append(new_row, ignore_index=True)
    # df.to_csv(ResultCSV, index=False)

    np.savez(SaveResultFilename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)

    print("--- %s seconds ---" % (time.time() - start_time))
c=np.load(SaveResultFilename+'.npz')
print('the average accuracy is'+format(np.average(c['saved_model_test_acc'])))
print(np.std(c['saved_model_test_acc']))

# print('the average test acc for 10 runs is'+ format(np.average(df['Ave_Test_Acc'])))


