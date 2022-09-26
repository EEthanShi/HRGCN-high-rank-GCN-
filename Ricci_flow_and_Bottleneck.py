#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:52:40 2022

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



#%%pre-prcoessing:

# def nodeAttack(x, ratio, normal=False):
#     if normal:
#             # x_new = x + torch.normal(0, ratio, size=(x.shape[0], x.shape[1])).to(device)
#             x_new = x + torch.normal(0, args.noiseLev, size=(x.shape[0], x.shape[1]))
#             x_new -= x_new.min() # make sure all values are non-negative
#     else:
#             mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
#             # mask = mask.to(device)
#             mask = mask
#             x -= mask.int()
#             x_new = (torch.abs(x)==1).double()
#     # return x_new.to(device)
#     return x_new.to(device)

def nodeAttack(x, ratio, std=0.1, normal=False):
    if normal:
            x_new= x + torch.normal(0, std, size=(x.shape[0], x.shape[1])).to(device)
            mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
            #mask = mask.to(device)
            polarity_mask = (torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < 0.5).to(device)
            polarity_mask = polarity_mask * 2
            polarity_mask = polarity_mask - 1
            x_new[mask]=x_new[mask]+polarity_mask[mask]
            #x_new=x_new.clamp(min=0)

    else:
            mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
            mask = mask.to(device)
            x -= mask.int()
            x_new = (torch.abs(x)==1).double()
            mask_2 = torch.DoubleTensor(x.shape[0], x.shape[1]).uniform_() < ratio
            mask_2 = mask_2.to(device)
            polarity_mask = torch.rand(x.shape).to(device)
            
            x_new[mask_2] = polarity_mask[mask_2].double()

    return x_new.to(device)



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
        #print('currrent dirichlet energy is'+ format(np.tra))
        
        x = F.dropout(x, self.dropout, training=self.training)
      
        x = self.gc2(x, adj)
        return F.log_softmax(x,dim=1)



#%% hpers 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
  
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='name of dataset (default: Cora): Cora, Citeseer, Pubmed,Coauthor Physics Coauthor CS Amazon Computer Amazon Photo')
 
    parser.add_argument('--reps', type=int, default=12,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.05, 
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=0.0001,  
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--nhid', type=int, default=96,   # for citation networks, this is 96,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.9,  
                        help='dropout probability (default: 0.7)')
    parser.add_argument('--Ollivier_alpha', type = float, default = 0.95, 
                        help='Ollivier_alpha value in Ollivier-Ricci Curvature (default: 0.7)')
    
    parser.add_argument('--noiseLev', type=float, default=0,   # values 0.05,0.15,0.25,0.50
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
dataset = Planetoid(root=rootname, name=dataname,split ='public')
#dataset = Coauthor(root=rootname, name=dataname)    
#dataset = Amazon(root=rootname, name=dataname)

dataset.transform = T.NormalizeFeatures()

#%%
data = dataset[0]
G = to_networkx(data, to_undirected=True)
#print('the rank of A is expected to be '+ format(matrix_rank(nx.adjacency_matrix(G).todense())))
num_nodes = dataset[0].x.shape[0]
nfeatures = dataset[0].x.shape[1]

print(f"Number of nodes in {dataname}:",num_nodes)
print(f"Number of Classes in {dataname}:", dataset.num_classes)
print(f"Number of Node Features in {dataname}:", dataset.num_node_features)


orc = OllivierRicci(G, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=args.num_nbr,exp_power=1) # should use 1 here since in the Olliviier compuation, we want to assign samilarity measure. 
orc.compute_ricci_curvature()  

G_orc = orc.G.copy()
rc = np.array(list(nx.get_edge_attributes(G_orc, 'ricciCurvature').values()))
adj = np.zeros((num_nodes, num_nodes))

adj_1 = nx.adjacency_matrix(G).todense()
adj_1 = sp.coo_matrix(adj_1) 
adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
adj_1 = normalize_adj(adj_1+sp.eye(adj_1.shape[0]))



features = data.x 
features = normalize_features(features.cpu())
features = torch.FloatTensor(np.array(features))
features_original = data.x

for n1, n2 in G_orc.edges():   # re-define the ricci curature as k/dij and k*dij to preserve the sign
    adj_1[n1, n2] = adj_1[n1, n2]*np.exp(-(orc.G[n1][n2]["ricciCurvature"])/(distance.euclidean(features_original[n1], features_original[n2])+np.random.uniform(0.1, 10**(-20))/10000))
    #adj_1[n1, n2] = adj_1[n1, n2]*np.exp(-(orc.G[n1][n2]["ricciCurvature"])*(distance.euclidean(features_original[n1], features_original[n2])+np.random.uniform(0.1, 10**(-20))/10000))
    adj_1[n2, n1] = adj_1[n1, n2]

adj_1= torch.FloatTensor(np.array(adj_1.todense()))


labels = data.y
data = data.to(device)


#%%

"""Using betness centality to measure bottlenecK"""

adj = nx.adjacency_matrix(G).todense()
adj= sp.coo_matrix(adj) 
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj= normalize_adj(adj + sp.eye(adj.shape[0]))
G_1 = nx.from_numpy_matrix(adj.todense())
between_central = nx.betweenness_centrality(G_1, normalized = False, endpoints = False,weight ='weight')  
print('the bottleneck value for Cora in GCN is '+ format(np.sum(list(between_central.values()))/num_nodes))

G_2 = nx.from_numpy_matrix(np.array(adj_1))
for e in G_2.edges():
  G[e[0]][e[1]]['ricciCurvature'] = adj_1[e[0],e[1]]
between_central_HRGCN = nx.betweenness_centrality(G_2, normalized = False, endpoints = False,weight = 'ricciCurvature') 
print('the bottleneck value for Cora in HRGCN is '+ format(np.sum(list(between_central_HRGCN.values()))/num_nodes))



#%%

#################################### This is the test for bottleneck using weight changes############################

""" in order to find out the weight changes of two layers network, we shall use two-hop neghbouthood. 
So, for the first hop, we use the most negative curvature; result as node index u,v
for the second hop, we use the largest positive curvature. resul  t as u,v' or v' u

Then we shwo the exp(-k tilde) smoothes the curvatures after two layers 
"""


adj_rc = np.zeros((num_nodes, num_nodes))
for n1, n2 in G_orc.edges():   # re-define the ricci curature as k/dij to preserve the sign
    adj_rc[n1, n2] = orc.G[n1][n2]["ricciCurvature"]
    adj_rc[n2, n1] = adj_rc[n1, n2]
# node_1 = np.where(adj_rc==np.min(adj_rc))[0][0] #find the node index with the min value of ricci curvature
# node_2 = np.where(adj_rc==np.min(adj_rc))[0][1]

node_1 = np.where(adj_rc==np.min(rc))[0][0] #find the node index with the min value of ricci curvature
node_2 = np.where(adj_rc==np.min(rc))[1][0]  # this is u,v with the most negative curvature 

node_2_prime = np.where(adj_rc[node_2]==sorted(adj_rc[node_2])[1])[0][0] # this is v'
print(np.min(rc))

A_hat = nx.adjacency_matrix(G).todense()
A_hat = sp.coo_matrix(A_hat) 
A_hat = A_hat + A_hat.T.multiply(A_hat.T > A_hat) - A_hat.multiply(A_hat.T > A_hat)
A_hat = normalize_adj(A_hat + sp.eye(A_hat.shape[0]))
A_hat_test = torch.linalg.matrix_power(torch.FloatTensor(A_hat.todense()), 2)

print('In A hat (power 2) the weight on this edge is'+format(A_hat_test[node_1,node_2_prime]))


adj_test = np.zeros((num_nodes, num_nodes))
for n1, n2 in G_orc.edges():   
    adj_test[n1, n2] = A_hat[n1, n2]*np.exp(-(orc.G[n1][n2]["ricciCurvature"])*(distance.euclidean(features_original[n1], features_original[n2])+np.random.uniform(0.1, 10**(-20))/10000))
    adj_test[n2, n1] = adj_test[n1, n2]
adj_test = torch.linalg.matrix_power(torch.FloatTensor(adj_test), 2)
print('In (exp -k dij times A hat) the weight on this edge is'+format(adj_test[node_1,node_2_prime]))

#%%

#########################  This is the curvature smoothing plots#####################



adj_1 = nx.adjacency_matrix(G).todense()
adj_1 = sp.coo_matrix(adj_1) 
adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
adj_1 = normalize_adj(adj_1 + sp.eye(adj_1.shape[0]))
adj_1 = adj_1.todense()

G_1 = nx.from_numpy_array(adj_1)
orc1 = OllivierRicci(G_1, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=args.num_nbr,exp_power=1) # should use 1 here since in the Olliviier compuation, we want to assign samilarity measure. 
orc1.compute_ricci_curvature()  
G1_orc = orc1.G.copy()
rc1 = np.array(list(nx.get_edge_attributes(G1_orc, 'ricciCurvature').values()))  # rc from initial A hat 



import plotly.express as px

fig = px.histogram(rc1,nbins=15,color_discrete_sequence=['teal'], 
                    title=' Histogram of Ricci Curvatures before HRGCN (Cora)')
fig.show()


for n1, n2 in G1_orc.edges():   # re-define the ricci curature as k/dij to preserve the sign
    adj_1[n1, n2] = np.exp(-G1_orc[n1][n2]["ricciCurvature"]*distance.euclidean(features[n1], features[n2]))
    adj_1[n2, n1] = adj_1[n1, n2]
adj_1
G_2 = nx.from_numpy_array(adj_1)
orc2 = OllivierRicci(G_2, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=args.num_nbr,exp_power=1) # should use 1 here since in the Olliviier compuation, we want to assign samilarity measure. 
orc2.compute_ricci_curvature()  
G2_orc = orc2.G.copy()
rc2 = np.array(list(nx.get_edge_attributes(G2_orc, 'ricciCurvature').values())) # this is the rc based on the weigth of A hat times exp(-rc1 tilde)
print(rc2)

fig = px.histogram(rc2,nbins=15,color_discrete_sequence=['teal'], 
                    title=' Histogram of Ricci Curvatures After HRGCN (Cora)')
fig.show()









