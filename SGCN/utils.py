################################################################################
# SGCN/utils.py
#
# Used to handle the command line arguments for the
# Signed Graph Convolutional Networks
#
# Author: Tyler Derr (derrtyle@msu.edu)
################################################################################


import argparse
from random import randint

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    
    parser = argparse.ArgumentParser(description= \
                                     """This is the code to run the SGCN.""")
    
    parser.add_argument('-n', '--network_file_name', type=str,
                        help="network dataset file name",
                        required=True)
                
    parser.add_argument('-f', '--feature_file_name', type=str,
                        help="feature dataset file name",
                        required=True)

    parser.add_argument('-v', '--val_network_file_name', type=str,
                        required=False, default=None)
    
    parser.add_argument('-l', '--learning_rate', type=float,
                        help="Learning rate for SGD", default=0.5)

    parser.add_argument('-s', '--random_seed', type=int,
                        help="seed for the random number generator",
                        default=randint(0, 2147483648))

    parser.add_argument('-e', '--embedding_size', type=int,
                        help="embedding length", default=32)

    parser.add_argument('-o', '--embedding_output_directory', type=str,
                        help="the directory name for where to store the embeddings",
                        required=True)

    parser.add_argument('-a', '--num_layers', type=int,
                        help="defines the number of aggregation layers",
                        default=2)#required=True)

    parser.add_argument('-b', '--batch_size', type=int,
                         help="while using SGD what is the minibatch size",
                         default=1000)

    parser.add_argument('-i', '--validation_interval', type=int,
                        help='while training what is validation interval in terms of minibatches',
                        default=100)

    parser.add_argument('-t', '--total_minibatches', type=int,
                        help='total number of minibatches to execute',
                        default=10000)

    parser.add_argument('-d', '--save_embeddings_interval', type=int,
                        help='how often to save the current embeddings',
                        default=100)

    parser.add_argument('-g', '--num_neighbors_sample', type=int,
                        help='how many neighbors to sample for aggregation',
                        default=250)

    parser.add_argument('-x', '--num_input_features', type=int,
                         help='how many input features to use from the input file',
                         default=None)#none means all

    parser.add_argument('-m', '--modify_input_features', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Activate the updating of the input features or not')

    parser.add_argument('--learn_decay', type=float,
                        help='gamma: learning rate decay rate every 100 step sizes',
                        default=0.75)

    parser.add_argument('--cross_entropy_weights',
                        type=lambda s: [float(item) for item in s.split('a')],
                        help='the weights for cross entropy part of the loss pos/neg/no links prediction',
                        default=None)
    
    parser.add_argument('--l2_weight_norm', type=float,
                         help='this controls the regularization on the network weights',
                         default=0.01)

    parser.add_argument('--loss2_regularization', type=float,
                        help='this is for controling the contribution of the second term',
                        default=5.0)
        
    args = parser.parse_args()
    print(args)
    
    args_dict = {}
    for arg in vars(args):
        args_dict[arg] = getattr(args, arg)
        
    if args_dict['cross_entropy_weights'] is None:
        #real default weights will be (pos, neg, no) link classes
        args_dict['cross_entropy_weights'] = [0.15, 0.80, 0.05]        

    if args_dict['cross_entropy_weights'] == [0,0,0]:
        args_dict['cross_entropy_weights'] = None

    return args_dict
