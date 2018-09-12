from trainer import *

def data_handler_random(inp_size, out_size):
	# create lists of inputs and labels
	inputs = []
	labels = []
	map = np.random.randn(out_size, inp_size)
	for i in range(inp_size):
		v = np.zeros([inp_size, 1])
		v[i][0] = 1
		inputs.append(deepcopy(v))
	
	labels.append(deepcopy(np.dot(map, v)))
	return inputs, labels

def data_handler_evol1D():
	# input = (x_, v_, w_)
	# output = (x, v)
	inputs = []
	labels = []
	dir = '/home/morsoni/dataset/evol1D'
	X = np.genfromtxt(dir + 'x.dat', delimiter = ',')
	V = np.genfromtxt(dir + 'v.dat', delimiter = ',')
	W = np.genfromtxt(dir + 'w.dat', delimiter = ',')
	assert X.shape == V.shape
	assert V.shape == W.shape
	
	n_realisation = 300
	X = X[:, n_realisation]
	V = V[:, n_realisation]
	W = W[:, n_realisation]
	for t in range(X.shape[0]-1): # runs through each t
		x_t = np.array([ [X[t]],[V[t]],[W[t]] ]) # column vec (X_, V_, W_)
		l_t = np.array([ [X[t+1]],[V[t+1]] ]) # column vec (X, V)
		inputs.append(x_t)
		labels.append(l_t)
	return inputs, labels
	
### Main
def main():
	
	n_training_examples = 4999
	inp_size = 3 #512 + 64
	out_size = 2 #512
	# inp_size = 100 and out_size = 80 : ~100s per epoch 
	lstm = LSTM(inp_size, out_size)
	inputs, labels = data_handler_evol1D()
	# forward_pass test
	out = lstm.forward_pass(inputs)
		
	# trainer test
	trainer = LSTM_Trainer(lstm)
	trainer.forward_backward_prop(inputs[0], labels[0])
	
	trainer.train(inputs, labels, learning_rate=0.0001, batch_size=5, n_epochs=1000)
	print(x_inputs[0])
	print(lstm.single_forward_pass(inputs[0]))
	print(labels[0])
	
if __name__ == "__main__":
	main()