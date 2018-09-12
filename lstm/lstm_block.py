from tools import *

class LSTM_Block:
	"""
	Follows implementation from Gers (2002): http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
	
	Follows notation proposed in "Generating Sequences With
	Recurrent Neural Networks", by Alex Graves: https://arxiv.org/pdf/1308.0850.pdf

	Notation: 
	a_t = a_{t}
	a_t_ = a_{t-1}
	

	"""
	def __init__(self, n_mem_cells: int, inp_size: int, out_size: int):
		
		self.inp_size = inp_size
		self.out_size = out_size
		self.n_mem_cells = n_mem_cells
		#self.mem_cells = [                ] #construct this list later
		
		# input gate
		self.w_xi = np.random.randn(out_size, inp_size)
		self.w_hi = np.random.randn(out_size, out_size)
		self.w_ci = np.random.randn(out_size, 1) # diagonal matrix, so make it into a column vector
		self.b_i = np.zeros([out_size, 1])
		# forget gate
		self.w_xf = np.random.randn(out_size, inp_size)
		self.w_hf = np.random.randn(out_size, out_size)
		self.w_cf = np.random.randn(out_size, 1) # diagonal matrix, so make it into a column vector
		self.b_f = np.zeros([out_size, 1])
		# output gate
		self.w_xo = np.random.randn(out_size, inp_size)
		self.w_ho = np.random.randn(out_size, out_size)
		self.w_co = np.random.randn(out_size, 1) # diagonal matrix, so make it into a column vector
		self.b_o = np.zeros([out_size, 1])
		# cell state
		self.w_xc = np.random.randn(out_size, inp_size)
		self.w_hc = np.random.randn(out_size, out_size)
		self.b_c = np.zeros([out_size, 1])
		
		self.c_list = [] # list containing this cell's c state for the last 2 time stamps
		self.h_list = [] # list containing this cell's h output for the last 2 time stamps

	def compute(self, x_t, returns:bool = False): 
		# Funcion H in the paper, computes the action of a cell by updating its c and h
		# input must be colum np.arrays
		assert x_t.shape == (self.inp_size, 1), "incompatible x_t format"

		if len(self.c_list) != 0 and len(self.h_list) != 0:
			c_t_ = self.c_list[-1] # last c computed
			h_t_ = self.h_list[-1] # last h computed 
		else:
			c_t_ = np.zeros([self.out_size, 1])
			h_t_ = np.zeros([self.out_size, 1])# should be zeros? 
			
		in_i = np.dot(self.w_xi, x_t) + np.dot(self.w_hi, h_t_) + self.w_ci * c_t_ + self.b_i
		i_t = sigmoid(in_i)
		in_f = np.dot(self.w_xf, x_t) + np.dot(self.w_hf, h_t_) + self.w_cf * c_t_ + self.b_f
		f_t = sigmoid(in_f)
		in_o = np.dot(self.w_xo, x_t) + np.dot(self.w_ho, h_t_) + self.w_co * c_t_ + self.b_o
		o_t = sigmoid(in_o)
		in_z = np.dot(self.w_xc, x_t) + np.dot(self.w_hc, h_t_) + self.b_c
		z_t = tanh(in_z) # cell's input
		c_t = f_t * c_t_ + i_t * z_t
		h_t = o_t * tanh(c_t) # no paper não tem esse tanh
		
		self.c_list.append(deepcopy(c_t))
		self.h_list.append(deepcopy(h_t))
		#self.c_list.append(c_t)
		#self.h_list.append(h_t)
		
		# only keeps the current values and the previous values
		if len(self.c_list)>2 and len(self.h_list)>2 :
			self.c_list.pop(0)
			self.h_list.pop(0)
			
		if returns:
			return in_i, in_f, in_o, in_z, c_t, h_t # these are more useful then i_t, f_t, ...,  for backprop
		
	def compute_(self, x_t, c_t_, h_t_): # testing which "compute" is better
		# Funcion H in the paper, computes the action of a cell by updating its c and h
		# input must be colum np.arrays
		assert x_t.shape == (self.inp_size, 1), "incompatible x_t format"
		assert c_t_.shape == (self.out_size, 1), "incompatible c_t_ format"
		assert h_t_.shape == (self.out_size, 1), "incompatible h_t_ format"

		i_t = sigmoid(np.dot(self.w_xi, x_t) + np.dot(self.w_hi, h_t_) + self.w_ci * c_t_ + self.b_i)
		f_t = sigmoid(np.dot(self.w_xf, x_t) + np.dot(self.w_hf, h_t_) + self.w_cf * c_t_ + self.b_f)
		c_t = f_t * c_t_ + i_t * tanh(np.dot(self.w_xc, x_t) + np.dot(self.w_hc, h_t_) + self.b_c)
		o_t = sigmoid(np.dot(self.w_xo, x_t) + np.dot(self.w_ho, h_t_) + self.w_co * c_t_ + self.b_o)
		h_t = o_t * tanh(c_t) # no paper não tem esse tanh
		
		self.c_list.append(c_t)
		self.h_list.append(h_t)
	
	def get_h(self, timestamp: int):
		return self.h_list[timestamp]
		
	def get_c(self, timestamp: int):
		return self.c_list[timestamp]
		
	def reset(self):
		self.c_list.clear()
		self.h_list.clear()