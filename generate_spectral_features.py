###########################################################################################
# generate_spectral_features_Sept29_2018.py
#
# Used to get the spectral embeddings that are later used as input
# into the Signed Graph Convolutional Networks (SGCNs).
# If other node features are existing they can be used instead of or in unison.
#
# Input 1: training_network_file_name.txt
#          Assumed format is:
#          numV numE
#          u_i u_j sign
#
# Input 2: size of the spectral embedding
#          i.e., the number of features for the SGCNs
#
# Author: Tyler Derr (derrtyle@msu.edu)
###########################################################################################

import pickle as p
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh, eigs
import numpy as np
import sys

###########################################################################################

def read_in_undirected_graph(file_name):
    with open(file_name) as fp:

        n,m = [int(val) for val in fp.readline().split()]
        #A = numpy.zeros((n,n), dtype=int)
        A = sps.dok_matrix((n,n), dtype=float)
        print('started reading...')
        # linenumber = 1
        for l in fp:
            # print(f'at line number {linenumber}, {l}')
            # linenumber += 1
            i,j,s = [int(val) for val in l.split()]
            if s>0:
                s = 1
            else:
                s = -1
            A[i-1,j-1] = s
            A[j-1,i-1] = s
    A = A.asformat('csr')
    return A

###########################################################################################

def get_D_from_A(A):
    print('getting D from A')
    D = sps.lil_matrix((A.shape[0],A.shape[1]), dtype=float)
    DD = np.dot(A,A.T)
    for i in range(A.shape[0]):
        D[i,i] = DD[i,i]
    D = D.asformat('csr')
    return D

###########################################################################################

if __name__ == "__main__":
    num_to_skip = 0
    training_network_file_name = sys.argv[1]
    A = read_in_undirected_graph(training_network_file_name)
    D = get_D_from_A(A)
    L = D - A
    n = A.shape[0]
    k_to_keep = int(sys.argv[2]) #number of features/spectral embedding size
    print('got the D and A, now getting the eigen vectors')
    val, vec = eigsh(L,k=k_to_keep,which='SM')
    #val, vec = eigsh(L,k=k_to_keep,which='LM',sigma=0.00001)
    print('got the eigen vectors')
    vec = vec[:,num_to_skip:]

    output = sys.argv[1][:-4] + '_features{}.pkl'.format(k_to_keep)
    print('Wrote the file to: {}'.format(output))
    p.dump(vec, open(output,'wb'))
