import torch
import torch.nn as nn
import torch_cluster
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, EdgePooling, BatchNorm

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nhid, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_channels=200, out_channels=5, kernel_size=3)
        self.conv_linear = nn.Sequential(nn.Linear(45, nhid))

        self.batch1 = BatchNorm(nhid)
        self.batch2 = BatchNorm(nhid)
        self.batch3 = BatchNorm(nhid)

        self.linear_in = nn.Sequential(nn.Linear(nfeat, nhid))

        self.linear_out = nn.Sequential(nn.Linear(2 * nhid, nclass))
        self.edge_pooling = EdgePooling(nhid)

    def forward(self, x, adj, features):

        x = self.linear_in(x.float())
        x = x.relu()
        x = self.dropout(x, training=self.training)


        x = self.gc1(x, adj.long(), features)
        x = self.batch1(x)
        x = x.relu()


        x = self.dropout(x, training=self.training)

        x = self.gc2(x, adj.long(), features)
        x = self.batch1(x)
        
        return x
