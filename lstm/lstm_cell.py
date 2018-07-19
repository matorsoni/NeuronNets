from functions import *

class LSTM_Cell:
	"""
	Follows notation proposed in "Generating Sequences With
	Recurrent Neural Networks", by Alex Graves: https://arxiv.org/pdf/1308.0850.pdf

	Notation: 
	a_t = a_{t}
	a_t_ = a_{t-1}
	"""
	def __init__(self, input_length: int):
		self.input_dim = input_length
		'''
		self.w_xi = np.zeros([input_length, input_length])
		self.w_hi = np.zeros([input_length, input_length])
		self.w_ci = np.zeros([input_length, input_length])
		self.b_i = np.zeros([input_length, 1])

		self.w_xf = np.zeros([input_length, input_length])
		self.w_hf = np.zeros([input_length, input_length])
		self.w_cf = np.zeros([input_length, input_length])
		self.b_f = np.zeros([input_length, 1])

		self.w_xc = np.zeros([input_length, input_length])
		self.w_hc = np.zeros([input_length, input_length])
		self.b_c = np.zeros([input_length, 1])

		self.w_xo = np.zeros([input_length, input_length])
		self.w_ho = np.zeros([input_length, input_length])
		self.w_co = np.zeros([input_length, input_length])
		self.b_o = np.zeros([input_length, 1])
		'''
		# pensar em iniciar os b's com valor alto, tipo 10
		self.w_xi = np.random.randn(input_length, input_length)
		self.w_hi = np.random.randn(input_length, input_length)
		self.w_ci = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_i = np.zeros([input_length, 1])

		self.w_xf = np.random.randn(input_length, input_length)
		self.w_hf = np.random.randn(input_length, input_length)
		self.w_cf = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_f = np.zeros([input_length, 1])

		self.w_xc = np.random.randn(input_length, input_length)
		self.w_hc = np.random.randn(input_length, input_length)
		self.b_c = np.zeros([input_length, 1])

		self.w_xo = np.random.randn(input_length, input_length)
		self.w_ho = np.random.randn(input_length, input_length)
		self.w_co = np.random.randn(input_length, 1) # diagonal matrix, so make it into a column vector
		self.b_o = np.zeros([input_length, 1])

	def compute(self, x_t, h_t_, c_t_):
		assert x_t.shape == (self.input_dim, 1), "incompatible x_t format"
		assert h_t_.shape == (self.input_dim, 1), "incompatible h_t_ format"
		assert c_t_.shape == (self.input_dim, 1), "incompatible c_t_ format"
		i_t = functions.sigmoid(np.dot(self.w_xi, x_t) + np.dot(self.w_hi, h_t_) + self.w_ci * c_t_ + self.b_i)
		f_t = functions.sigmoid(np.dot(self.w_xf, x_t) + np.dot(self.w_hf, h_t_) + self.w_cf * c_t_ + self.b_f)
		c_t = f_t * c_t_ + i_t * functions.tanh(np.dot(self.w_xc, x_t) + np.dot(self.w_hc, h_t_) + self.b_c)
		o_t = functions.sigmoid(np.dot(self.w_xo, x_t) + np.dot(self.w_ho, h_t_) + self.w_co * c_t_ + self.b_o)
		h_t = o_t * functions.tanh(c_t)

		return h_t, c_t
		

dim = 10000
ls = LSTM_Cell(dim)
x_t = np.random.randn(dim, 1)
h_t_ = np.random.randn(dim, 1)
c_t_ = np.random.randn(dim, 1)

h_t, c_t = ls.compute(x_t, h_t_, c_t_)

print(h_t.shape, c_t.shape)
print(h_t[:5], c_t[:5])