################################################################################
# SGCN/model.py
#
# Used to create the Signed Graph Convolutional Networks
#
# Author: Tyler Derr (derrtyle@msu.edu)
# Note: This is based on the Reference PyTorch GraphSAGE Implementation.
#       https://github.com/williamleif/graphsage-simple
################################################################################

from __future__ import print_function
from collections import defaultdict
from random import randint
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn import init
from torch.autograd import Variable

from SGCN.utils import get_arguments
from SGCN.encoders import LayerEncoder
from SGCN.aggregators import FirstLayerAggregator, NonFirstLayerAggregator

import numpy as np
import os
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################

class SignedGCN(nn.Module):
    """
    Based on the SupervisedGraphSage Class
    """

    def __init__(self, num_nodes, final_embedding_size, enc,
                 cross_entropy_weights, loss2_regularization):
        super(SignedGCN, self).__init__()
        self.num_nodes = num_nodes
        self.enc = enc
        self.margin = 0.1
        self.loss2_regularization = loss2_regularization
        if cross_entropy_weights is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(cross_entropy_weights)
            )

        self.distance_fn = torch.nn.PairwiseDistance(p=2)

        self.final_embedding_size = final_embedding_size
        self.weight = nn.Parameter(torch.FloatTensor(final_embedding_size, 2*enc.embed_dim))
        self.W_to_class_dim = nn.Parameter(torch.FloatTensor(2*final_embedding_size, 3))
        #because 3 classes (i.e., +,?,-)

        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.W_to_class_dim)

    ################################################################################

    def forward(self, nodes):
        embeds_bal, embeds_unbal = self.enc(nodes)
        if __debug__:
            print('embeds_bal/unbal sizes: ', embeds_bal.size(), embeds_unbal.size())

        combined_embedding = torch.cat([embeds_bal.t(), embeds_unbal.t()], dim=1)
        if __debug__:
            print('combined embedding size: ', combined_embedding.size())

        final_embedding = F.tanh(self.weight.mm(combined_embedding.t())) #sigmoid, relu
        return(final_embedding.t())

    ################################################################################

    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg):
        max_node_index = self.num_nodes - 1
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []

        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        for i in center_nodes:
            #if no links then we can ignore
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0,max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0,max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)

            need_samples = 2 # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0,max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    #got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_map = {}
        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node:i for i,node in enumerate(all_nodes_list)}

        final_embedding = self.forward(all_nodes_list)

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys)

        #now that we have the mapped indices and final embeddings we can get the loss
        avg_loss = self.loss_fn(
            torch.mm(
                torch.cat(
                    (final_embedding[i_indices_mapped],
                     final_embedding[j_indices_mapped]),
                    1),
                self.W_to_class_dim
            ),
            ys
        )

        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        avg_loss2 = torch.mean(
            torch.max(
                torch.zeros(len(i_loss2)),
                self.distance_fn(
                    final_embedding[i_loss2],
                    final_embedding[pos_no_loss2]
                )**2
                - self.distance_fn(
                    final_embedding[i_loss2],
                    final_embedding[no_neg_loss2]
                )**2
            )
        )

        return avg_loss + self.loss2_regularization*avg_loss2

    ############################################################################

    def save_embedding(self, args, epoch, embedding_directory):
        all_nodes_list = list(range(self.num_nodes))
        #no map necessary for ids as we are using all nodes
        final_embedding = self.forward(all_nodes_list)
        final_embedding = final_embedding.detach().numpy()

        if args['cross_entropy_weights'] == None:
            xent_str = '0a0a0'
        else:
            xent_str = 'a'.join([str(val) for val in args['cross_entropy_weights']])

        #learning_rate, embed size, batch_size, num_layers, num_neighbors_sample, num_input_features
        embedding_directory = '{}lr{}_es{}_bs{}_nl{}_nns{}_nif{}_ld{}_xw{}_lw{}_loss{}/'.format(
            embedding_directory, args['learning_rate'], args['embedding_size'],
            args['batch_size'], args['num_layers'], args['num_neighbors_sample'],
            args['num_input_features']+1, args['learn_decay'], xent_str,
            args['l2_weight_norm'], args['loss2_regularization'])

        if not os.path.exists(embedding_directory):
                os.makedirs(embedding_directory)
        output = embedding_directory + 'embedding_epoch' + str(epoch) + '.pkl'
        pickle.dump(final_embedding, open(output,'wb'))

    ############################################################################

    def validation(self, adj_lists_pos, adj_lists_neg,
                   val_adj_lists_pos, val_adj_lists_neg):
        all_nodes_list = list(range(self.num_nodes))
        #no map necessary for ids as we are using all nodes
        final_embedding = self.forward(all_nodes_list)
        final_embedding = final_embedding.detach().numpy()
        #training dataset
        X_train = []
        y_train = []
        X_val = []
        y_val_true = []
        for i in range(self.num_nodes):
            for j in adj_lists_pos[i]:
                temp = np.append(final_embedding[i],final_embedding[j])
                #if np.isnan(np.min(temp)):
                #    print('error pos train' , i,j)
                X_train.append(temp)
                y_train.append(1)

            for j in adj_lists_neg[i]:
                temp = np.append(final_embedding[i],final_embedding[j])
                #if np.isnan(np.min(temp)):
                #    print('error neg train' , i,j)
                X_train.append(temp)
                y_train.append(-1)

            for j in val_adj_lists_pos[i]:
                temp = np.append(final_embedding[i],final_embedding[j])
                #if np.isnan(np.min(temp)):
                #    print('error pos val' , i,j)
                X_val.append(temp)
                y_val_true.append(1)

            for j in val_adj_lists_neg[i]:
                temp = np.append(final_embedding[i],final_embedding[j])
                #if np.isnan(np.min(temp)):
                #    print('error neg val' , i,j)
                X_val.append(temp)
                y_val_true.append(-1)

        y_train = np.asarray(y_train)
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_val_true = np.asarray(y_val_true)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train,y_train)
        y_val_pred = model.predict(X_val)

        auc = roc_auc_score(y_val_true, y_val_pred)
        f1 = f1_score(y_val_true, y_val_pred)

        return auc, f1



################################################################################
def read_in_undirected_network(file_name):
    links = {}
    with open(file_name) as fp:
        n,m = [int(val) for val in fp.readline().split()]
        for l in fp:
            rater, rated, rating = [int(val) for val in l.split()]
            if rating > 0:
                rating = 1
            else:
                assert(rating != 0)
                rating = -1
            edge1 = (rater,rated)
            edge2 = (rated,rater)
            if edge1 not in links:
                links[edge1] = rating
                links[edge2] = rating
            elif links[edge1] == rating:#we had it before and it was the same
                pass
            else:#we had it before and now it's a different value
              #set to negative
                links[edge1] = -1
                links[edge2] = -1

    adj_lists_pos = defaultdict(set)
    adj_lists_neg = defaultdict(set)

    for (i,j),s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
        else:
            adj_lists_neg[i].add(j)

    return n, adj_lists_pos, adj_lists_neg

################################################################################
def read_in_feature_data(feature_file_name, num_input_features):
    feat_data = pickle.load(open(feature_file_name, 'rb'))
    if num_input_features is not None:
        #we perform a shrinking as to which features we are using
        feat_data = feat_data[:,:num_input_features]

    num_nodes, num_feats = feat_data.shape

    max_vals = feat_data.max(axis=0)
    min_vals = feat_data.min(axis=0)

    #standardizing the input features
    feat_data = StandardScaler().fit_transform(feat_data)#.T).T

    return num_feats, feat_data

################################################################################
def load_data(network_file_name, feature_file_name, val_network_file_name, num_input_features):
    num_nodes, adj_lists_pos, adj_lists_neg = read_in_undirected_network(network_file_name)

    num_feats, feat_data = read_in_feature_data(feature_file_name, num_input_features)

    if val_network_file_name is not None:
        val_num_nodes, val_adj_lists_pos, val_adj_lists_neg = \
                                            read_in_undirected_network(val_network_file_name)
    else:
        val_num_nodes, val_adj_lists_pos, val_adj_lists_neg = None, None, None

    return num_nodes, adj_lists_pos, adj_lists_neg, \
        num_feats, feat_data, val_adj_lists_pos, val_adj_lists_neg

################################################################################
def run(args, cuda=False):

    embed_size = args['embedding_size']
    num_nodes, adj_lists_pos, adj_lists_neg, \
        num_feats, feat_data, \
        val_adj_lists_pos, val_adj_lists_neg = \
        load_data(args['network_file_name'],
                  args['feature_file_name'],
                  args['val_network_file_name'],
                  args['num_input_features'])

    if args['num_input_features'] is None:
        args['num_input_features'] = num_feats

    features = nn.Embedding(num_nodes, num_feats)

    if args['modify_input_features']:
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
    else:
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    if cuda:
        features.cuda()

    self_separate = True
    if self_separate:
        sep_or_merged = 1
    else:
        sep_or_merged = 0

    num_neighbors_sample = args['num_neighbors_sample']


    ##################################################################
    #this is for two layer
    ##################################################################
    if args['num_layers'] == 2:
        agg1 = FirstLayerAggregator(1, features, only_layer=False, cuda=cuda)
        enc1 = LayerEncoder(1, (sep_or_merged+1)*num_feats, embed_size, adj_lists_pos, adj_lists_neg, agg1,
                            num_sample = num_neighbors_sample, cuda=cuda, last_layer = False)

        agg2 = NonFirstLayerAggregator(2, lambda nodes : enc1(nodes), cuda=cuda)
        enc2 = LayerEncoder(2, (sep_or_merged + 2)*enc1.embed_dim, embed_size, adj_lists_pos, adj_lists_neg, agg2,
                            num_sample = num_neighbors_sample, base_model=enc1, cuda=cuda,
                            last_layer = True)
        agg3 = NonFirstLayerAggregator(2, lambda nodes : enc2(nodes), cuda=cuda)
        enc3 = LayerEncoder(2, (sep_or_merged + 2)*enc2.embed_dim, embed_size, adj_lists_pos, adj_lists_neg, agg3,
                            num_sample = num_neighbors_sample, base_model=enc2, cuda=cuda,
                            last_layer = True)

        signedGCN = SignedGCN(num_nodes, embed_size, enc3,
                              args['cross_entropy_weights'], args['loss2_regularization'])
    else:
        raise NotImplementedError('we advise using 2 layers... see code to use additional layers')

    if cuda:
        signedGCN.cuda()

    train = list(np.random.permutation(list(range(0,num_nodes)))) #list(rand_indices[:])
    total_batches = args['total_minibatches']
    batch_size = args['batch_size']
    batch_start = 0
    batch_end = batch_size
    save_embeddings_interval = args['save_embeddings_interval']
    if val_adj_lists_pos is not None:
        validation_interval = args['validation_interval']
    else:
        validation_interval = total_batches + 1 #i.e., never run the validation

    optimizer = torch.optim.SGD(
        filter(lambda p : p.requires_grad, signedGCN.parameters()),
        lr=args['learning_rate'], weight_decay=args['l2_weight_norm'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args['learn_decay'])


    epoch_losses = []
    batch_losses = []
    val_auc_f1 = []
    epoch_loss = 0
    epoch = 1
    for batch in range(total_batches):

        #signedGCN.train()
        #get minibatch
        if batch_end > len(train):
            print('epoch {} loss: '.format(epoch), epoch_loss)
            epoch += 1
            epoch_losses.append(epoch_loss)
            epoch_loss = 0
            batch_start = 0
            batch_end = batch_size
            random.shuffle(train)
        batch_center_nodes = train[batch_start:batch_end]
        batch_start = batch_end
        batch_end += batch_size


        #forward step
        optimizer.zero_grad()
        loss = signedGCN.loss(batch_center_nodes,adj_lists_pos,adj_lists_neg)


        #backward step
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_loss = loss.item()
        batch_losses.append(batch_loss)
        epoch_loss += batch_loss


        if (batch + 1) % validation_interval == 0:
            #signedGCN.eval()
            optimizer.zero_grad()
            auc,f1 = signedGCN.validation(adj_lists_pos, adj_lists_neg,
                                          val_adj_lists_pos, val_adj_lists_neg)
            print(batch, ' validation sign prediction (auc,f1) :', auc,'\t',f1)
            sys.stdout.flush()
            val_auc_f1.append((auc,f1))


        if (batch + 1) % save_embeddings_interval == 0:
            #signedGCN.eval()
            signedGCN.save_embedding(args, batch, args['embedding_output_directory'])


    #store the network here if wanted for future use and also save the final embeddings
    #signedGCN.eval()
    signedGCN.save_embedding(args, batch, args['embedding_output_directory'])
print('hey')
if __name__ == "__main__":
    #Get arguments and set the random seed
    print("there")
    cuda = False
    args = get_arguments()
    seed = args['random_seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    #Run the model
    run(args, cuda=cuda)
