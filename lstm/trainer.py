from lstm import *

class d_Cell:
	"""
	Container class for dC_t/d_W of each cell.
	"""
	def __init__(self, input length: int):
		# gates
		# C is independent of the Output gate
		self.d_w_xi = np.zeros([input_length, input_length])
		self.d_w_hi = np.zeros([input_length, input_length])
		self.d_w_ci = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_i = np.zeros([input_length, 1])

		self.d_w_xf = np.zeros([input_length, input_length])
		self.d_w_hf = np.zeros([input_length, input_length])
		self.d_w_cf = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_f = np.zeros([input_length, 1])
		
		# external weights
		self.d_w_xc = np.zeros([input_length, input_length])
		self.d_w_hc = np.zeros([input_length, input_length])
		self.d_b_c = np.zeros([input_length, 1])

class d_Cost:
	"""
	Container class for dE_t/d_W for each W, aka the total derivative at timestamp t. 
	"""
	def __init__(self, input length: int):
		# gates
		self.d_w_xi = np.zeros([input_length, input_length])
		self.d_w_hi = np.zeros([input_length, input_length])
		self.d_w_ci = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_i = np.zeros([input_length, 1])

		self.d_w_xf = np.zeros([input_length, input_length])
		self.d_w_hf = np.zeros([input_length, input_length])
		self.d_w_cf = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_f = np.zeros([input_length, 1])

		self.d_w_xo = np.zeros([input_length, input_length])
		self.d_w_ho = np.zeros([input_length, input_length])
		self.d_w_co = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_o = np.zeros([input_length, 1])
		
		self.d_w_xc = np.zeros([input_length, input_length])
		self.d_w_hc = np.zeros([input_length, input_length])
		self.d_b_c = np.zeros([input_length, 1])
		
		# external weights
		self.d_W_hy = np.zeros([lstm.input_length, lstm.input_length])
		self.d_b_y = np.zeros([lstm.input_length, 1])

class LSTM_Trainer:
	def __init__(self, lstm : LSTM):
		self.lstm = lstm
		self.d_cell = d_Cell(lstm.input_length)
		self.d_cost = d_Cost(lstm.input_length)
	
	def forward_backward_prop(self, cost_function:str = 'MSE', w_type:str, x_t, l_t):
		# forward prop:
		in_i, in_f, in_o, in_z, c_t, h_t = self.lstm.block.compute(x_t, True)
		y_t = sigmoid(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y)
		
		# backprop:
		if cost_function == 'MSE':
			delta_h = np.dot( row(y_t-l_t), self.lstm.d_Y(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y) * self.lstm.w_hy )
		
		## output gate
		
	
	def train(self, x_inputs:list, learning_rate, batch_size:int, epochs:int):
		pass
		
		
		
		
		
		
		
		