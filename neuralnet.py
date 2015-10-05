import numpy as np
import scipy.io
import math
import csv
def tanh(x):
	return np.tanh(x)
def sigmoid(x):
	return 1/(1+np.exp(-x))
def makelabel(i):
	label = np.matrix(np.zeros(10))
	label[0, i] = 1
	return label
def mean_squared_error(prediction, label):
	return 0.5 * np.sum(np.square(label - prediction))
def mean_squared_prime(prediction, label):
	return -(label - prediction)
def cross_entropy(prediction, label):
	return -1 * np.sum(np.multiply(label, np.log(prediction)) + np.multiply((1-label), np.log(1-prediction)))
def cross_entropy_prime(prediction, label):
	return -(np.divide(label, prediction)) + (np.divide((1 - label), (1 - prediction)))
class Neuralnet:
	def __init__(self, sizes, learningrate, lossfunction, maxiterations):
		self.sizes = sizes #list of sizes, not including bias terms
		self.lossfunction = lossfunction
		self.maxiterations = maxiterations
		self.learningrate = learningrate
		self.layers = []
		#initialize hiddenlayer objects for each layer. The input does not have a hidden layer object.
		for i in range(1, len(self.sizes)-1):
			self.layers.append(Hiddenlayer(tanh, sizes[i-1], sizes[i], False))
		self.layers.append(Hiddenlayer(sigmoid, sizes[-2], sizes[-1], True))

	def train(self, dataset, labels):
		for i in range(self.maxiterations):
			# Section for plotting training error and classification error every 2000 iterations
			# if i % 2000 == 0:
			# 	totalloss = 0 
			# 	iterations = len(dataset)
			# 	errors = 0
			# 	for k in range(iterations):
			# 		instance = dataset[k,:]
			# 		label = labels[k]
			# 		output = self.forwardprop(instance)
			# 		if np.argmax(output) != np.argmax(label):
			# 			errors += 1
			# 		loss = self.lossfunction(output, label)
			# 		totalloss += loss
			# 	avgloss = totalloss / float(iterations)
			# 	avgerror = errors / float(iterations)
			# 	print str(i) + ', ' + str(avgloss) + ', ' + str(avgerror)
			j = i
			i = i % len(dataset) 
			assert dataset[i, :].shape == (1, 785)
			instance = dataset[i, :]
			label = labels[i]
			assert label.shape == (1, 10)
			output = self.forwardprop(instance)
			loss = self.lossfunction(output, label)
			if self.lossfunction == mean_squared_error:
				updates = self.backwardprop(instance, label, mean_squared_prime, output)
			elif self.lossfunction == cross_entropy:
				updates = self.backwardprop(instance, label, cross_entropy_prime, output)
			outputlayer = self.layers[1]
			hiddenlayer = self.layers[0]
			outputlayer.weights = outputlayer.weights - ((self.learningrate)/(1+0.001*j))*updates[1]
			hiddenlayer.weights = hiddenlayer.weights - ((self.learningrate)/(1+0.001*j))*updates[0]
	def predict(self, dataset):
		predictions = []
		for i in range(len(dataset)): 
			assert dataset[i, :].shape == (1, 785)
			instance = dataset[i, :]
			label = labels[i]
			assert label.shape == (1, 10)
			output = self.forwardprop(instance)
			predictions.append(output)
		return predictions
	def forwardprop(self, data):
		#train_data is given as a matrix of training instances, with each row vector being an instance. (60000 x 784)
		currentinput = data
		for layer in self.layers: 
			currentinput = layer.compute_output(currentinput)
		output = self.layers[-1].X
		return output
	def backwardprop(self, input, label, gradientfunc, prediction):
		# calculate delta for output layer
		outputlayer = self.layers[1]
		hiddenlayer = self.layers[0]

		if gradientfunc == mean_squared_prime:
			outputlayer.delta = np.multiply(gradientfunc(prediction, label), np.multiply(outputlayer.X, (1-outputlayer.X)))
		elif gradientfunc == cross_entropy_prime:
			outputlayer.delta = outputlayer.X - label

		hiddenlayer.delta = np.multiply(1-np.square(hiddenlayer.X), (outputlayer.weights*outputlayer.delta.transpose()).transpose())
		#one additional delta for the bias we dont need. Remove first element
		hiddenlayer.delta = hiddenlayer.delta[:, 1:]

		outupdate = hiddenlayer.X.transpose() * outputlayer.delta
		hidupdate = input.transpose() * hiddenlayer.delta
		return [hidupdate, outupdate]
	def compute_gradient(self, k, e, input, label):
		layer = self.layers[k]
		gradient = np.zeros([layer.n_in +1, layer.n_out])
		for i in range(len(layer.weights)):
			for j in range(len(layer.weights[0])):
				layer.weights[i, j] += e
				out1 = self.forwardprop(input)
				layer.weights[i, j] -= 2*e
				out2 = self.forwardprop(input)
				layer.weights[i, j] += e
				loss1 = mean_squared_error(out1, label)
				loss2 = mean_squared_error(out2, label)
				gradient[i, j] = (loss1 - loss2) / (2 * e)
		return gradient

class Hiddenlayer: 
	def __init__(self, activation, n_in, n_out, outputlayer):
		self.X = None #row vector of node values, with n_out + 1 (bias term) elements, except the output layer
		self.n_in = n_in # d(l-1) number of units in layer l-1 excluding bias. 
		self.n_out = n_out # d(l), number of units in layer l excluding bias
		self.weights = 	np.matrix(np.random.randn(self.n_in+1, self.n_out) * 0.001, dtype=np.float64) # n_in by n_out matrix of weights from layer l-1 to l, weights[i, j]
		self.weights[0,:] = 0 #set bias weights to 0 
		self.delta = None # local gradient we store on foward pass
		self.activation = activation
		self.outputlayer = outputlayer
	#computes output Sj of this layer given input Xi. Sj = input*weights.
	def compute_output(self, currentinput):
		self.S = currentinput * self.weights
		self.X = self.activation(self.S)
		if not self.outputlayer:
			self.X = np.insert(self.X, 0, values=1, axis=1)
		return self.X


def compute_error(predictions, labels):
	errors = 0
	for i in range(len(predictions)):
		if np.argmax(predictions[i]) != np.argmax(labels[i]):
			errors += 1
	return errors

def write_file(predictions):
	with open('test.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, dialect='excel')
		for item in predictions:
			spamwriter.writerow([item])
def makeclass(predictions):
	classes = []
	for prediction in predictions:
		classes.append(np.argmax(prediction))
	return classes


train_mat = scipy.io.loadmat('digit-dataset/train.mat')
test_mat = scipy.io.loadmat('digit-dataset/test.mat')
train_data = np.array(train_mat['train_images'], dtype=np.float64)
test_data = np.array(test_mat['test_images'], dtype=np.float64)

train_data = np.reshape(train_data, (784, 60000))
train_data = train_data.transpose()

test_data = np.reshape(test_data, (784, 10000))
test_data = test_data.transpose()

train_data -= np.matrix(np.mean(train_data, axis=1)).transpose()
train_data /= np.matrix(np.std(train_data, axis=1)).transpose()
train_data = np.insert(train_data, 0, values=1, axis=1)

test_data -= np.matrix(np.mean(test_data, axis=1)).transpose()
test_data /= np.matrix(np.std(test_data, axis=1)).transpose()
test_data = np.insert(test_data, 0, values=1, axis=1)

randomizer = [i for i in range(0, 60000)]
np.random.shuffle(randomizer)
train_data = train_data[randomizer, :]

train_labels = np.array(train_mat['train_labels'])
train_labels = train_labels[randomizer]
vector_labels = []
for label in train_labels:
	vector_labels.append(makelabel(label))
vector_labels = np.array(vector_labels)

nnet = Neuralnet([784, 200, 10], 0.1, mean_squared_error, 180000)
nnet2 = Neuralnet([784, 200, 10], 0.1, cross_entropy, 180000)

train = np.matrix(train_data[0:60000,:], dtype=np.float64)
labels = np.matrix(vector_labels[0:60000])

test = np.matrix(test_data, dtype=np.float64)
val = np.matrix(train_data[59001:60000, :], dtype=np.float64)
val_labels = np.matrix(vector_labels[59001:60000])









# if __name__ == '__main__':
# 	main()