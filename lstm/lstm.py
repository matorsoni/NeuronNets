from lstm_cell import *

class LSTM(object):
	def __init__(self, n_in, n_out, n_fwd_layers, n_hid_layers, input_length):
		'''

		'''
		# assert n_in == n_out? se sim, tirar n_out dos args
		# assert n_in == n_hid_layers? se sim, tirar dos args
		assert n_fwd_layers >= n_in, "Incompatible numbers of inputs and forward layers"
		assert n_fwd_layers >= n_out, "Incompatible numbers of outputs and forward layers"
		assert n_hid_layers >= 1, "Hidden layers >= 1"
		self.n_in = n_in
		self.n_out = n_out
		self.n_hid_layers = n_hid_layers
		self.n_fwd_layers = n_fwd_layers
		
		self.cells = [ [LSTM_Cell(input_length) for x in range(n_fwd_layers)] for y in range(n_hid_layers) ]
		# conectar as cells
		# sera que os pointers pras cells output são necessarias? se pa nao, pois pra fazer o forward feed basta usar 
		# um loop for, sem precisar dos pointers

	def __str__(self):
		string = ''
		for i in range(self.n_fwd_layers-self.n_out):
			string += '      ' # 6 spaces
		for i in range(self.n_out):
			string += 'out   ' # 3 spaces
		string += '\n'

		for y in range(self.n_hid_layers):
			for x in range(self.n_fwd_layers):
				string += 'cell  '
			string += '\n'

		for i in range(self.n_in):
			string += 'in    ' # 4 spaces
		for i in range(self.n_fwd_layers-self.n_in):
			string += '      ' # 6 spaces
		string += '\n'

		return string

a = LSTM(1, 2, 4, 2, input_length=2)
print(a.cells)