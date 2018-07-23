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

		# linking all the cells
		for y in range(n_hid_layers):
			for x in range(n_fwd_layers-1): # link left and right cells in the same layer
				self.cells[y][x].set_pointer2cell(self.cells[y][x+1], 'right')
				self.cells[y][n_fwd_layers - 1 - x].set_pointer2cell(self.cells[y][n_fwd_layers - 2 - x], 'left')
			
			if y>0: # link current layer to the layer below it
				for x in range(n_fwd_layers):
					self.cells[y][x].set_pointer2cell(self.cells[y-1][x], 'down')
					self.cells[y-1][x].set_pointer2cell(self.cells[y][x], 'up')
	
	def forward_pass(self, input:list):
		# input = list[x_0, x_1, x_2, ...]
		assert len(input) == self.n_in, 'Incompatible number of inputs'
		


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