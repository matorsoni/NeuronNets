from lstm_cell import *

class LSTM(object):
	def __init__(self, n_in, n_out, n_hid_layers, n_fwd_layer):
		'''

		'''
		# assert n_in == n_out? se sim, tirar n_out dos args
		# assert n_in == n_hid_layers? se sim, tirar dos args
		assert n_fwd_layer >= n_in, "Incompatible numbers of inputs and forward layers"
		self.n_in = n_in
		self.n_out = n_out
		self.n_hid_layers = n_hid_layers
		self. n_fwd_layer = n_fwd_layer
		
		