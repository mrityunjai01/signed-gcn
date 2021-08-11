most code extended from the sample implementation by Tyler Derr.


################################################################################
Step 1) Ensure appropriate undirected signed network dataset format
################################################################################

	The format of the file first line should be of the following form:
	num_nodes num_edges, these values will be used for the adjacency matrix, so be sure they are correct.

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

	Use 64 for dim_size, make sure you run this on a bash shell, not the default windows one.
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
