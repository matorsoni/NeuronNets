from functions import *

from copy import deepcopy

class Gate: # One input, output and forget gates for each LSTM Bloc
	pass 

class Mem_Cell:
	def __init__(self, input_length: int):
		pass
	pass

class LSTM_Block:
	"""
	Follows implementation from Gers (2002): http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
	
	Follows notation proposed in "Generating Sequences With
	Recurrent Neural Networks", by Alex Graves: https://arxiv.org/pdf/1308.0850.pdf

	Notation: 
	a_t = a_{t}
	a_t_ = a_{t-1}
	"""
	def __init__(self, n_mem_cells: int, input_length: int, input_block = None, output_block = None):
		# pointers to neighbour cells 
		#assert len(input_block) == 1, "Must have 1 input cell, can be None"
		#assert len(output_block) == 1, "Must have 1 output cell, can be None"
		
		self.down_block = input_block
		self.up_block = output_block
		
		self.input_length = input_length
		self.n_mem_cells = n_mem_cells
		#self.mem_cells = [                ] #construir essa lista mais tarde
		
		# input gate
		self.w_xi = np.random.randn(input_length, input_length)
		self.w_hi = np.random.randn(input_length, input_length)
		self.w_ci = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_i = np.zeros([input_length, 1])
		# forget gate
		self.w_xf = np.random.randn(input_length, input_length)
		self.w_hf = np.random.randn(input_length, input_length)
		self.w_cf = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_f = np.zeros([input_length, 1])
		# output gate
		self.w_xo = np.random.randn(input_length, input_length)
		self.w_ho = np.random.randn(input_length, input_length)
		self.w_co = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_o = np.zeros([input_length, 1])
		# cell state
		self.w_xc = np.random.randn(input_length, input_length)
		self.w_hc = np.random.randn(input_length, input_length)
		self.b_c = np.zeros([input_length, 1])
		
		self.c_list = [] # list containing this cell's c state for every time stamp
		self.h_list = [] # list containing this cell's h output for every time stamp
		#self.c = np.zeros([input_length, 1]) # this cell's state variable 
		#self.h = np.zeros([input_length, 1]) # this cell's output variable 

	def compute(self, x_t): 
		# Funcion H in the paper, computes the action of a cell by updating its c and h
		# input must be colum np.arrays
		assert x_t.shape == (self.input_length, 1), "incompatible x_t format"

		if len(self.c_list) != 0 and len(self.h_list) != 0:
			c_t_ = self.c_list[-1] # last c computed
			h_t_ = self.h_list[-1] # last h computed 
		else:
			c_t_ = np.zeros([self.input_length, 1])
			h_t_ = np.zeros([self.input_length, 1])# should be zeros? 

		i_t = functions.sigmoid(np.dot(self.w_xi, x_t) + np.dot(self.w_hi, h_t_) + self.w_ci * c_t_ + self.b_i)
		f_t = functions.sigmoid(np.dot(self.w_xf, x_t) + np.dot(self.w_hf, h_t_) + self.w_cf * c_t_ + self.b_f)
		o_t = functions.sigmoid(np.dot(self.w_xo, x_t) + np.dot(self.w_ho, h_t_) + self.w_co * c_t_ + self.b_o)
		z_t = functions.tanh(np.dot(self.w_xc, x_t) + np.dot(self.w_hc, h_t_) + self.b_c) # cell's input
		c_t = f_t * c_t_ + i_t * z_t
		h_t = o_t * functions.tanh(c_t) # no paper não tem esse tanh
		
		self.c_list.append(deepcopy(c_t))
		self.h_list.append(deepcopy(h_t))
		#self.c = c_t
		#self.h = h_t
		
	def compute2(self, x_t, c_t_, h_t_): # testing which "compute" is better
		# Funcion H in the paper, computes the action of a cell by updating its c and h
		# input must be colum np.arrays
		assert x_t.shape == (self.input_length, 1), "incompatible x_t format"
		assert c_t_.shape == (self.input_length, 1), "incompatible c_t_ format"
		assert h_t_.shape == (self.input_length, 1), "incompatible h_t_ format"

		i_t = functions.sigmoid(np.dot(self.w_xi, x_t) + np.dot(self.w_hi, h_t_) + self.w_ci * c_t_ + self.b_i)
		f_t = functions.sigmoid(np.dot(self.w_xf, x_t) + np.dot(self.w_hf, h_t_) + self.w_cf * c_t_ + self.b_f)
		c_t = f_t * c_t_ + i_t * functions.tanh(np.dot(self.w_xc, x_t) + np.dot(self.w_hc, h_t_) + self.b_c)
		o_t = functions.sigmoid(np.dot(self.w_xo, x_t) + np.dot(self.w_ho, h_t_) + self.w_co * c_t_ + self.b_o)
		h_t = o_t * functions.tanh(c_t) # no paper não tem esse tanh
		
		self.c_list.append(c_t)
		self.h_list.append(h_t)
	
	def get_h(self, timestamp: int):
		return self.h_list[timestamp]
	def get_c(self, timestamp: int):
		return self.c_list[timestamp]
		
	def set_pointer2cell(self, cell, orientation: str):
		if orientation == 'down':
			self.down_cell = cell
		elif orientation == 'up':
			self.up_cell = cell
			
	def __str__(self):
		# ver isso depois...
		s = '      ' # 6 spaces
		return s