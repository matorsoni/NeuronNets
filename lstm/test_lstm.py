from lstm import *

def main():

	T = 3
	n_out = 1
	n_layers = 4
	n_mem_cells = 1
	input_length = 10
	lstm = LSTM(T, n_out, n_layers, n_mem_cells, input_length)
	print(lstm)

	inputs = []
	for i in range(T):
		inputs.append(np.zeros([input_length, 1]))
	
	out = lstm.forward_pass(inputs) # todos outputs tao iguais, mudar isso
	for o in out:
		print(o)
		
	#print(lstm.cells[0].h_list)
	#print(lstm.cells[1].h_list)

if __name__ == "__main__":
	main()
