################################################################################
# SGCN/encoders.py
#
# Used to define the aggregation methods for the SGCNs
#
# Author: Tyler Derr (derrtyle@msu.edu)
# Note: This is based on the Reference PyTorch GraphSAGE Implementation.
#       https://github.com/williamleif/graphsage-simple
################################################################################

from __future__ import print_function
from torch.nn import init

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerEncoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    The non-first hidden layers will will need to merge across the pos and neg
    """
    def __init__(self, id,  feature_dim, 
                 embed_dim, adj_lists_pos, adj_lists_neg,
                 aggregator, num_sample=10,
                 base_model=None, cuda=False, last_layer=False): 
        super(LayerEncoder, self).__init__()
        
        self.id = id
        self.last_layer = last_layer        
        self.feat_dim = feature_dim
        self.adj_lists_pos = adj_lists_pos
        self.adj_lists_neg = adj_lists_neg
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight_bal = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim))
        self.weight_unbal = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim))
        init.xavier_uniform_(self.weight_bal)
        init.xavier_uniform_(self.weight_unbal)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #Get the balanced/unbalanced neighbor feats aggregated
        if self.last_layer:
            bal_neigh_feats,unbal_neigh_feats = self.aggregator.forward(nodes,
                                                [self.adj_lists_pos[node] for node in nodes],
                                                [self.adj_lists_neg[node] for node in nodes],
                                                self.num_sample)
        else:
            bal_neigh_feats,unbal_neigh_feats = self.aggregator.forward(nodes,
                                                [self.adj_lists_pos[node.item()] for node in nodes],
                                                [self.adj_lists_neg[node.item()] for node in nodes],
                                                self.num_sample)

        mapped_bal_neigh_feats = F.tanh(self.weight_bal.mm(bal_neigh_feats.t()))
        mapped_unbal_neigh_feats = F.tanh(self.weight_unbal.mm(unbal_neigh_feats.t()))
        
        return mapped_bal_neigh_feats, mapped_unbal_neigh_feats
