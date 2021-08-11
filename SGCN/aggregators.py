################################################################################
# SGCN/aggregators.py
#
# Used to define the aggregation methods for the SGCNs
#
# Author: Tyler Derr (derrtyle@msu.edu)
# Note: This is based on the Reference PyTorch GraphSAGE Implementation.
#       https://github.com/williamleif/graphsage-simple
################################################################################

from torch.autograd import Variable

import random
import torch
import torch.nn as nn

"""
Set of modules for aggregating embeddings of neighbors.
"""
class NonFirstLayerAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    using the logic of positive and negative links from balanced and
    unbalanced paths
    """
    def __init__(self, id, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(NonFirstLayerAggregator, self).__init__()
        self.id = id
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs_pos, to_neighs_neg, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs_pos --- list of sets, each set is the set of positive neighbors for node in batch
        to_neighs_neg --- list of sets, each set is the set of negative neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs_pos = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs_pos]
            samp_neighs_neg = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs_neg]
        else:
            samp_neighs_pos = to_neighs_pos
            samp_neighs_neg = to_neighs_neg

        if __debug__:
            print (nodes, ' nodes agg')
            print ('agg samp_neighs_pos before ', samp_neighs_pos)

        self_nodes = [set([nodes[i]]) for i, samp_neigh in enumerate(nodes)]

        if __debug__:
            print ('agg samp_neighs_pos after ', samp_neighs_pos)

        #easier just to keep them together and have two separate masks
        unique_nodes_list = list(set.union(*samp_neighs_pos).union(*samp_neighs_neg).union(*self_nodes))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        mask_pos = Variable(torch.zeros(len(samp_neighs_pos), len(unique_nodes)))
        mask_neg = Variable(torch.zeros(len(samp_neighs_neg), len(unique_nodes)))
        mask_self = Variable(torch.zeros(len(self_nodes), len(unique_nodes)))

        column_indices_pos = [unique_nodes[n] for samp_neigh in samp_neighs_pos for n in samp_neigh]
        column_indices_neg = [unique_nodes[n] for samp_neigh in samp_neighs_neg for n in samp_neigh]
        column_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]

        row_indices_pos = [i for i in range(len(samp_neighs_pos)) for j in range(len(samp_neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(samp_neighs_neg)) for j in range(len(samp_neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for j in range(len(self_nodes[i]))]

        if __debug__:
            print (len(row_indices_pos), len(row_indices_neg), len(column_indices_pos), len(column_indices_neg), ' row col lens (pos/neg)')

        mask_pos[row_indices_pos, column_indices_pos] = 1
        mask_neg[row_indices_neg, column_indices_neg] = 1
        mask_self[row_indices_self, column_indices_self] = 1

        if self.cuda:
            mask_pos = mask_pos.cuda()
            mask_neg = mask_neg.cuda()
            mask_self = mask_self.cuda()

        num_neigh_pos = mask_pos.sum(1, keepdim=True)
        num_neigh_neg = mask_neg.sum(1, keepdim=True)
        num_self_nodes = mask_self.sum(1, keepdim=True)

        #find the divide by 0 and change to 1, to keep value to 0 after division with no errors
        for i in range(num_neigh_pos.shape[0]):
            if num_neigh_pos[i].item() == 0:
                num_neigh_pos[i] = 1
        for i in range(num_neigh_neg.shape[0]):
            if num_neigh_neg[i].item() == 0:
                num_neigh_neg[i] = 1
        for i in range(num_self_nodes.shape[0]):
            if num_self_nodes[i].item() == 0:
                num_self_nodes[i] = 1

        mask_pos = mask_pos.div(num_neigh_pos)
        mask_neg = mask_neg.div(num_neigh_neg)
        mask_self = mask_self.div(num_self_nodes)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        embed_matrix_bal = embed_matrix[0].t()
        embed_matrix_unbal = embed_matrix[1].t()

        to_feats_bal = torch.cat([mask_pos.mm(embed_matrix_bal),
                                  mask_neg.mm(embed_matrix_unbal),
                                  mask_self.mm(embed_matrix_bal)],
                                 dim=1)
        to_feats_unbal = torch.cat([mask_pos.mm(embed_matrix_unbal),
                                    mask_neg.mm(embed_matrix_bal),
                                    mask_self.mm(embed_matrix_unbal)],
                                 dim=1)

        return to_feats_bal, to_feats_unbal

class FirstLayerAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, id, features,  only_layer, cuda=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(FirstLayerAggregator, self).__init__()
        self.id = id
        self.features = features
        self.cuda = cuda
        self.only_layer = only_layer

    def forward(self, nodes, to_neighs_pos, to_neighs_neg, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs_pos --- list of sets, each set is the set of positive neighbors for node in batch
        to_neighs_neg --- list of sets, each set is the set of negative neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs_pos = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs_pos]
            samp_neighs_neg = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs_neg]
        else:
            samp_neighs_pos = to_neighs_pos
            samp_neighs_neg = to_neighs_neg

        if __debug__:
            print ('aggregator ', self.id, ' ...')
            print (nodes, ' nodes agg')
            print ('agg samp_neighs_pos before ', samp_neighs_pos)

        if self.only_layer:
            self_nodes = [set([nodes[i]]) for i, samp_neigh in enumerate(nodes)]
        else:
            self_nodes = [{nodes[i].item()} for i, samp_neigh in enumerate(nodes)]

        if __debug__:
            print ('agg samp_neighs_pos after ', samp_neighs_pos)

        #easier just to keep them together and have two separate masks
        unique_nodes_list = list(set.union(*samp_neighs_pos).union(*samp_neighs_neg).union(*self_nodes))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        mask_pos = Variable(torch.zeros(len(samp_neighs_pos), len(unique_nodes)))
        mask_neg = Variable(torch.zeros(len(samp_neighs_neg), len(unique_nodes)))
        mask_self = Variable(torch.zeros(len(self_nodes), len(unique_nodes)))

        column_indices_pos = [unique_nodes[n] for samp_neigh in samp_neighs_pos for n in samp_neigh]
        column_indices_neg = [unique_nodes[n] for samp_neigh in samp_neighs_neg for n in samp_neigh]
        column_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]

        row_indices_pos = [i for i in range(len(samp_neighs_pos)) for j in range(len(samp_neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(samp_neighs_neg)) for j in range(len(samp_neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for j in range(len(self_nodes[i]))]

        # if __debug__:
            # print (len(row_indices), len(column_indices), ' row col lens')

        mask_pos[row_indices_pos, column_indices_pos] = 1
        mask_neg[row_indices_neg, column_indices_neg] = 1
        mask_self[row_indices_self, column_indices_self] = 1

        if self.cuda:
            mask_pos = mask_pos.cuda()
            mask_neg = mask_neg.cuda()
            mask_self = mask_self.cuda()

        num_neigh_pos = mask_pos.sum(1, keepdim=True)
        num_neigh_neg = mask_neg.sum(1, keepdim=True)
        num_self_nodes = mask_self.sum(1, keepdim=True)

        #find the divide by 0 and change to 1, to keep value to 0 after division with no errors
        for i in range(num_neigh_pos.shape[0]):
            if num_neigh_pos[i].item() == 0:
                num_neigh_pos[i] = 1
        for i in range(num_neigh_neg.shape[0]):
            if num_neigh_neg[i].item() == 0:
                num_neigh_neg[i] = 1
        for i in range(num_self_nodes.shape[0]):
            if num_self_nodes[i].item() == 0:
                num_self_nodes[i] = 1

        mask_pos = mask_pos.div(num_neigh_pos)
        mask_neg = mask_neg.div(num_neigh_neg)
        mask_self = mask_self.div(num_self_nodes)

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        to_feats_bal = torch.cat([mask_pos.mm(embed_matrix),
                                  mask_self.mm(embed_matrix)],
                                 dim=1)
        to_feats_unbal = torch.cat([mask_neg.mm(embed_matrix),
                                    mask_self.mm(embed_matrix)],
                                   dim=1)

        if __debug__:
            print (self.id, ' aggregators sizes feats bal/unbal: ', to_feats_bal.size(), to_feats_unbal.size())
            print (to_feats_bal, to_feats_unbal)
        return to_feats_bal, to_feats_unbal
