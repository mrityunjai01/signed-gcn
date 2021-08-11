################################################################################
# get_auc_f1_from_embedding.py
#
# Used to obtain the AUC and F1 from from the SSE or SGCN methods on the
# sign prediction task using their saved embeddings.
#
# Author: Tyler Derr (derrtyle@msu.edu)
################################################################################

from __future__ import print_function
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pickle as p
import sys


################################################################################
def read_in_feature_data(feature_file_name):
    feat_data = p.load(open(feature_file_name, 'rb'))
    num_nodes, num_feats = feat_data.shape
    print('num feats loaded from embedding data:', num_feats)
    print('num nodes loaded from embedding data:', num_nodes)
    return feat_data

################################################################################
def read_in_undirected_network(file_name):
    adj_lists_pos = defaultdict(set)
    adj_lists_neg = defaultdict(set)
    with open(file_name) as fp:
        n,m = [int(val) for val in fp.readline().split()]
        for l in fp:
            i,j,s = [int(val) for val in l.split()]

            if s > 0:
                s = 1
                adj_lists_pos[i].add(j)
                adj_lists_pos[j].add(i)
            else:
                assert(s != 0)
                s = -1
                adj_lists_neg[i].add(j)
                adj_lists_neg[j].add(i)

    return n, adj_lists_pos, adj_lists_neg

################################################################################

train_network_file_name = sys.argv[1]
feature_file_name = sys.argv[2]
test_network_file_name = sys.argv[3]
embed_size = int(sys.argv[4])

#get embedding
final_embedding = read_in_feature_data(feature_file_name)

#get train/test network
n, adj_lists_pos_train, adj_lists_neg_train = read_in_undirected_network(train_network_file_name)
n, adj_lists_pos_test, adj_lists_neg_test = read_in_undirected_network(test_network_file_name)

X_train = []
y_train = []
X_test = []
y_test_true = []
for i in range(n):
    for j in adj_lists_pos_train[i]:
        X_train.append(np.append(final_embedding[i][:embed_size],final_embedding[j][:embed_size]))
        y_train.append(1)

    for j in adj_lists_neg_train[i]:
        X_train.append(np.append(final_embedding[i][:embed_size],final_embedding[j][:embed_size]))
        y_train.append(-1)

    for j in adj_lists_pos_test[i]:
        X_test.append(np.append(final_embedding[i][:embed_size],final_embedding[j][:embed_size]))
        y_test_true.append(1)

    for j in adj_lists_neg_test[i]:
        X_test.append(np.append(final_embedding[i][:embed_size],final_embedding[j][:embed_size]))
        y_test_true.append(-1)

y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_test_true = np.asarray(y_test_true)
model = LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)

y_test_pred = model.predict(X_test)

auc = roc_auc_score(y_test_true, y_test_pred)
f1 = f1_score(y_test_true, y_test_pred)
print('auc, f1: ', auc, f1)
