from trainer import *
import csv

def write_list(l, filename):
	with open(filename, 'w') as wfile:
		wr = csv.writer(wfile)
		wr.writerow(l)

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
	dir = '/home/morsoni/dataset/evol1D/input/'
	X = np.genfromtxt(dir + 'x.csv', delimiter = ',')
	V = np.genfromtxt(dir + 'v.csv', delimiter = ',')
	W = np.genfromtxt(dir + 'w.csv', delimiter = ',')
	assert X.shape == V.shape
	assert V.shape == W.shape
	
	n_realisation = 700
	X = X[:, n_realisation]
	V = V[:, n_realisation]
	W = W[:, n_realisation]
	for t in range(X.shape[0]-1): # runs through each t
		x_t = np.array([ [X[t]],[V[t]],[W[t]] ]) # column vec (X_, V_, W_)
		l_t = np.array([ [X[t+1]],[V[t+1]] ]) # column vec (X, V)
		inputs.append(x_t)
		labels.append(l_t)
	return inputs, labels

def test_model_1D(lstm ,inputs, labels):
	save_dir='/home/morsoni/dataset/evol1D/output/'
	#output_list = []
	position_input_list = [x[0][0] for x in inputs]; position_input_list.pop()
	vel_input_list = [x[1][0] for x in inputs]; vel_input_list.pop()

	position_output_list=[]
	vel_output_list=[]
	error_list = []
	x_t = inputs[0]
	for t in range(len(labels)-1):
		y_t=lstm.single_forward_pass(x_t)
		x_out = y_t[0][0]
		v_out = y_t[1][0]
		w = inputs[t+1][2][0]
		err = np.dot(row(y_t-labels[t]), y_t-labels[t])[0]/lstm.inp_size
		x_t = col( np.array([x_out, v_out, w]) )
		
		error_list.append(err)
		position_output_list.append(x_out)
		vel_output_list.append(v_out)
		#output_list.append(y_t)
	#return position_input_list, position_output_list, error_list
	write_list(position_input_list, save_dir+'pos_in.csv')
	write_list(position_output_list, save_dir+'pos_out.csv')
	write_list(vel_input_list, save_dir+'vel_in.csv')
	write_list(vel_output_list, save_dir+'vel_out.csv')
	write_list(error_list, save_dir+'error.csv')	
	
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
	
	trainer.train(inputs, labels, learning_rate=0.001, batch_size=10, n_epochs=10)
	#position_input_list, position_output_list, error_list = test_model_1D(lstm, inputs, labels)
	test_model_1D(lstm ,inputs, labels)

	save_model(lstm, 'model1')


if __name__ == "__main__":
	main()
