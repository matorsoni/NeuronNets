from lstm_cell import *

class LSTM(object):
	def __init__(self, T, n_out, n_layers, input_length):
		'''

		'''
		assert T >= n_out, "Incompatible numbers of outputs and forward layers"
		assert n_layers >= 1, "Hidden layers >= 1"
		self.T = T
		self.n_out = n_out
		self.n_layers = n_layers
		
		self.cells = [LSTM_Cell(input_length) for n in range(n_layers)]

		# linking all the cells
		for n in range(n_layers):
			if n>0: # link current layer to the layer below it	
				self.cells[n].set_pointer2cell(self.cells[n-1], 'down')
				self.cells[n-1].set_pointer2cell(self.cells[n], 'up')

		self.W_ih = [np.random.randn(input_length, input_length) for i in range(n_layers)] # weights from input to cells n
		self.W_hh = [np.random.randn(input_length, input_length) for i in range(n_layers)] # weights from layers t-1 to t, cells n to n
		self.W_h_h = [np.random.randn(input_length, input_length) for i in range(n_layers-1)] # weights from layer t to t, cell n-1 to n
		self.b_h = [np.random.randn(input_length, 1) for i in range(n_layers)] # biases in each cell in layer t
		self.W_hy = [np.random.randn(input_length, input_length) for i in range(n_layers)] # weights from cells n to output y
		self.b_y = np.random.randn(input_length, 1)
	
	def forward_pass(self, x_inputs:list):
		# input = list[x_0, x_1, x_2, ...], x_i = column np.array
		assert len(x_inputs) == self.T, 'Incompatible number of x_inputs'
		output_list = []
		for t in range(self.T):
		
			for n in range(self.n_layers):
				if n==0: # if it's the first hidden layer there is no n-1
					if t==0: # if it's the first input there is no t-1
						self.cells[n].update(np.dot(self.W_ih[n], x_inputs[t]) + self.b_h[n])
					else:
						self.cells[n].update(
							np.dot(self.W_ih[n], x_inputs[t]) + np.dot(self.W_hh[n], self.cells[n].get_h(-1)) + 
							self.b_h[n]	)
				else:
					if t==0:
						self.cells[n].update(np.dot(self.W_ih[n], x_inputs[t]) + 
							np.dot(self.W_h_h[n-1], self.cells[n-1].h) + self.b_h[n])
					else:
						self.cells[n].update(
							np.dot(self.W_ih[n], x_inputs[t]) + np.dot(self.W_hh[n], self.cells[n].get_h(-1)) + 
							np.dot(self.W_h_h[n-1], self.cells[n-1].get_h(-1)) + self.b_h[n]
							)
			
			y = self.b_y
			print(t)
			for n in range(self.n_layers):
				y += np.dot(self.W_hy[n], self.cells[n].get_h(-1))
			output_list.append(y)  # Actually it is Y(y), the output layer function !!!! (what is this, exactly?)
		
		return output_list # esse outputlist ta sempre com T elementos....
		
	def __str__(self):
		string = ''
		for i in range(self.T-self.n_out):
			string += '      ' # 6 spaces
		for i in range(self.n_out):
			string += 'out   ' # 3 spaces
		string += '\n'

		for n in range(self.n_layers):
			for t in range(self.T):
				string += 'cell  '
			string += '\n'

		for i in range(self.T):
			string += 'in    ' # 4 spaces
		string += '\n'

		return string