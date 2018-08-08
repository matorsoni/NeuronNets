from lstm import *

def main():

	'''
	# lstm_cell test
	dim = 10000
	ls = LSTM_Cell(dim)
	x_t = np.random.randn(dim, 1)
	h_t_ = np.random.randn(dim, 1)
	c_t_ = np.random.randn(dim, 1)

	h_t, c_t = ls.compute(x_t, h_t_, c_t_)

	print(h_t.shape, c_t.shape)
	print(h_t[:5], c_t[:5])
	'''
	n_in = 3
	n_out = 1
	n_hid_layers = 4
	input_length = 10
	lstm = LSTM(3, 1, 4, 10)
	print(lstm)

	inputs = []
	for i in range(n_in):
		inputs.append(np.zeros([input_length, 1]))
	
	out = lstm.forward_pass(inputs) # todos outputs tao iguais, mudar isso
	for o in out:
		print(o)
		
	#print(lstm.cells[0].h_list)
	#print(lstm.cells[1].h_list)

if __name__ == "__main__":
	main()
