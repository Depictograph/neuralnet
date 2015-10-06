README 

The NeuralNet class has the following constructor: 
	# Sizes is a list of layer sizes, including the input and output layers, each excluding the bias term. Loss function currently supports mean_squared_error and cross_entropy, both of which use their corresponding derivatives. 
	nnet = NeuralNet([sizes], learning_rate, loss_function, max_iterations) 

and the following methods: 
	# Takes in the set of all training examples and corresponding labels. Dataset is a training_instances by feature_size + 1 vector. The first column of this matrix is expect to be all 1’s for the bias term. Labels is a training_instances by 10 matrix, with each label being a 1 in 10 mapping binary mapping with the index corresponding to the digit class. This method performs up to max_iterations of back-propagation on the training set
	NeuralNet.train(dataset, labels) 

	#Takes in a dataset and outputs the predictions of the dataset as a matrix of label vectors, as described above. This method does not perform back-propagation 
	NeuralNet.predict(dataset)

running the code with:
	python -i neuralnet.py
will preprocess the training and test data, and result in matrices, train, labels, and test. and construct a Neuralnet called nnet using MSE, and a Neuralnet called nnet2 using cross entropy error. 

In the python interpreter, the neural nets can be trained using:
	nnet.train(train, labels) 
	predictions = nnet.predict(test) 

Helper functions are provided for writing to files, finding the argmax of each label, and computing error. 
