import os
from os import path
import sys
from random import randint

#filenames
dataset = sys.argv[1] #'bitcoin_otc'
network_file_name = f'{dataset}_train.txt'
feature_file_name = f'{dataset}_train_features128.pkl' #although we only use 64
val_network_file_name = f'{dataset}_val.txt' #can be replaced with test after tuning for a new dataset
embedding_output_directory = f'embeddings/{dataset}_val/' #this directory will be created later

#parameters
random_seed = randint(0,2**32 - 1)
validation_interval = 100
total_minibatches = 10000
save_embeddings_interval = 100
modify_input_features = 'True'

learning_rate = 0.5
embedding_size = 32 #for both the friend and foe embedding (so this size x2 for overall final embedding size)
num_layer = 2
batch_size = 1000 #100 worked better for epinions
num_neighbors_sample = 250
num_input_feature = 64
learn_decay = 0.75 #0.95 worked better for epinions
xent_weight = '0.15a0.8a0.05' #positive, negative, neutral (since they were unbalanced and different importance)
l2_weight = 0.01 #0.0 worked better for slashdot
loss2_reg = 5

output_direc = '{}_test/lr{}_es{}_bs{}_nl{}_nns{}_nif{}_mif{}_ld{}_xw{}_l2{}_loss{}'.format(dataset,
               learning_rate, embedding_size, batch_size, num_layer,
               num_neighbors_sample, num_input_feature, modify_input_features,
               learn_decay,xent_weight,l2_weight,loss2_reg)

cmd = '"python.exe -m SGCN.model --network_file_name {} --feature_file_name {} --val_network_file_name {} --learning_rate {} --random_seed {} --embedding_size {} --embedding_output_directory {} --num_layers {} --batch_size {} --validation_interval {} --total_minibatches {} --save_embeddings_interval {} --num_neighbors_sample {} --num_input_features {} --modify_input_features {} --learn_decay {} --cross_entropy_weights {} --l2_weight_norm {} --loss2_regularization {} 1> {} 2>&1"'.format(
network_file_name, feature_file_name, val_network_file_name, learning_rate, random_seed, embedding_size, embedding_output_directory, num_layer, batch_size, validation_interval, total_minibatches, save_embeddings_interval, num_neighbors_sample, num_input_feature, modify_input_features, learn_decay, xent_weight, l2_weight, loss2_reg, output_direc)

if __debug__:
    print('Command that would be executed:\n')
    print(cmd)
    print('\nto execute the above program rerun not in debug mode using the below\n')
    print('$ python -O run_default_general_parameters.py dataset_name\n')

else:
    print(cmd)
    os.system(cmd)
    print('hrere')
