#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:39:17 2022

@author: daishi
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from my_SAGP_networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from torch_geometric.utils import get_laplacian, to_networkx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.spatial import distance
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--Ollivier_alpha', type = float, default = 0.95, # 调参结果0.9 和0.95 
                        help='Ollivier_alpha value in Ollivier-Ricci Curvature (default: 0.7)')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])



train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        G = to_networkx(data, to_undirected=True)
        print('number of nodes in this graph is '+ format(G.number_of_nodes()))
        orc = OllivierRicci(G, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=100000)
        orc.compute_ricci_curvature()  
        G_orc = orc.G.copy()
        adj  = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
        features = data.x
        for n1, n2 in G_orc.edges():
            adj[n1, n2] = np.exp(-(orc.G[n1][n2]["ricciCurvature"])/(distance.euclidean(features[n1], features[n2])+np.random.uniform(0.1, 10**(-20))/10000))
            adj[n2, n1] = adj[n1, n2]
        
        data = data.to(args.device)
        out = model(data,adj)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        G = to_networkx(data, to_undirected=True)
        print('number of nodes in this graph is '+ format(G.number_of_nodes()))
        orc = OllivierRicci(G, alpha=args.Ollivier_alpha, verbose="INFO",shortest_path="all_pairs", nbr_topk=100000)
        orc.compute_ricci_curvature()  
        G_orc = orc.G.copy()
        adj  = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
        features = data.x
        for n1, n2 in G_orc.edges():
            adj[n1, n2] = np.exp(-(orc.G[n1][n2]["ricciCurvature"])/(distance.euclidean(features[n1], features[n2])+np.random.uniform(0.1, 10**(-20))/10000))
            adj[n2, n1] = adj[n1, n2]
        data = data.to(args.device)
        out = model(data,adj)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss = test(model,test_loader)
print("Test accuarcy:{}".fotmat(test_acc))
