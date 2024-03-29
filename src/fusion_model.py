import torch
import torch.nn as nn
from vision_transformer import ViT
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, EdgePooling, BatchNorm

class FusionModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth, image_size, patch_size, heads, dropout):
        super(FusionModel, self).__init__()

        self.gcn = GraphConv(nhid, nhid)
        self.vit  = ViT( image_size = image_size * 5, patch_size = patch_size, 
                           num_classes = nclass, dim = nhid, depth =depth, 
                           heads = heads, mlp_dim = nhid, 
                           dropout = dropout, emb_dropout = dropout, channels=nfeat )


        self.batch_conv1 = nn.BatchNorm2d(5)
        self.batch_conv2 = nn.BatchNorm2d(5)

        self.batch1 = BatchNorm(nhid)
        self.batch2 = BatchNorm(nhid)
        self.batch3 = BatchNorm(nhid)

        self.linear_in = nn.Sequential(nn.Linear(nfeat, nhid))

        self.linear_out = nn.Sequential(nn.Linear(2 * nhid ,  nclass))

        self.edge_pooling = EdgePooling(nhid)

    def forward(self, x, adj, features, slices):

        x = self.linear_in(x.float())
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)


        x_gcn = self.gcn(x, adj.long(), features)

        x_vit = self.vit(slices.transpose(1, 3).float())
        
        # merge
        x = torch.cat([x_gcn, x_vit], dim=1)

        x = self.linear_out(x)
        
        return x