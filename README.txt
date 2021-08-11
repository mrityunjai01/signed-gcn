################################################################################
# README.txt
#
# Used to describe how to run the Signed Graph Convolutional Networks code.
#
# Author: Tyler Derr (derrtyle@msu.edu)
# Note: Some code extended from the Reference PyTorch GraphSAGE Implementation.
#       https://github.com/williamleif/graphsage-simple
#
# Version information:
#  $ python
#  Python 2.7.12 (default, Dec  4 2017, 14:50:18)
#  [GCC 5.4.0 20160609] on linux2
#  Type "help", "copyright", "credits" or "license" for more information.
#  >>> import torch
#  >>> import sklearn
#  >>> import scipy
#  >>> import numpy
#  >>> torch.__version__
#  '0.4.0'
#  >>> sklearn.__version__
#  '0.19.1'
#  >>> scipy.__version__
#  '1.0.0'
#  >>> numpy.__version__
#  '1.14.2'
#
################################################################################

################################################################################
Step 1) Ensure appropriate undirected signed network dataset format
################################################################################

	The format of the file first line should be of the following form:
	num_nodes num_edges

	The format of the remaining lines should be as follows:
	node_i node_j sign

	where num_nodes and num_edges are the number of nodes and edges in the
	network, respectively, while node_i and node_j are indices in [0,num_nodes-1]
	used to represent a undirected link between the two nodes and sign is either
	a 1 or -1 (representing a positive or negative link between the two nodes,
	respectively).

################################################################################
Step 2) Obtain the spectral embedding by running generate_spectral_features.py
################################################################################

	Note: We recommend obtaining 128, since we can later just limit to the
	first 64 (which we used in our experiments). This will output a .pkl file
	containing the embedding/features.

	How to run this file:
	$ python generate_spectral_features.py training_data_file_name dim_size
	
	where training_data_file_name is of the same form of the original
	undirected signed network file mentioned above, but should not include the
	testing/validation links, and dim_size is the number of features.
     
################################################################################
Step 3) Run the Signed Graph Convolutional Network (SGCN) code
################################################################################

	$ python run_default_general_parameters.py dataset_name

	where dataset_name is the name of the dataset. 

	Note that this file assumes you have the following namining convension:
	dataset_name_train.txt, dataset_name_val.txt, dataset_name_train_features128.pkl
	which the later will be consistent when generated if the first is.
	
	This will show you the way to run the SGCN code. If you want to run with
	the somewhat general default parameters you can then run:

	$ python -O run_default_general_parameters.py dataset_name

	Note: This code has been used not using a GPU, however on line 461
	of the SGCN/model.py file it can be changed to set cuda = True
	instead of having cuda = False. We do not currently guarantee
	the code working with cuda, but will ensure this works for the
	next version. 

################################################################################
Step 4) Get test AUC/F1 from the spectral and SGCN method embeddings
################################################################################

	$ python get_auc_f1_from_embedding.py network_train network_train_embeddings network_test embedding_size

	where the parameters are the training network file (as described in step 1),
	the embeddings generated from either the SSE or SGCN method, the testing
	network file (as described in step 1), and size of embeddings

	Note for the SSE method the file might contain 128 for example, but you
	can specify to use the first 64.
