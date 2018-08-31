import numpy as np
from copy import deepcopy
import pickle

### pickle ###
dir = 'models/'
ext = 'pkl'
def save_model(model_object, file_name: str):
	file = dir + file_name
	if file_name.partition('.')[2] == '':
		file += '.' + ext
	pickle.dump(model_object, open(file, 'wb'))
	
def load_model(file_name: str):
	file = dir + file_name
	if file_name.partition('.')[2] == '':
		file += '.' + ext
	return pickle.load(open(file, 'rb'))
##############

### functions ###
def sigmoid(z):
	return 1./(1.+np.exp(-z))
	
def d_sigmoid(z):
	return sigmoid(z)*(1.-sigmoid(z))

def relu(z):
	return 0. if z<=0. else z

def d_relu(z):
	return 0. if z<=0. else 1.
	
def tanh(z):
	return np.tanh(z)
		
def d_tanh(z):
	return 1.-(np.tanh(z) * np.tanh(z)) 

def choose(func_name = 'sigmoid'):
	if func_name == 'sigmoid':
		return sigmoid, d_sigmoid
	elif func_name == 'relu':
		return relu, d_relu
	elif func_name == 'tanh':
		return tanh, d_tanh
	else:
		print("Invalid function name:" + func_name)
		return 0;
################

### matrix algebra ###
def col(v):
	return v.reshape(v.size,1)
	
def row(v):
	return v.reshape(v.size)

def vec2full_mat(v, n_rows: int):
	# constructs a matrix whose rows are v
	if v.shape == (v.size,1):
		v = row(v)
	return np.array([v for i in range(n_rows)])

def vec2zero_mat(v, n_rows:int, k: int):
	# constructs a matrix whose rows are 0 except for the k-th row, which is v
	assert k<n_rows
	if v.shape == (v.size,1):
		v = row(v)
	m = np.zeros([n_rows, v.size])
	m[k] = v
	return m
	
def vec2diag_mat(v):
	# just a redimensioned diagonal matrix whose diag is the vector v 
	return np.diagflat(v).reshape(v.size, v.size, 1)

def vec2ten(v, n_rows):
	# constructs a tensor whose k-th matrix is a zero matrix with v as the k-th row	
	return np.array([vec2zero_mat(v, n_rows, i) for i in range(n_rows)])
		
	
def vec_dot_ten(vec, ten):
	# constructs a new tensor with the same dimensions as ten
	assert vec.size == ten.shape[0]
	return np.array([vec[k]*ten[k] for k in range(vec.size)])
#########################
	
### miscellaneous ###
def select_and_pop(l:list):
	# chooses a random value in the list and pop it
	random_index = np.random.randint(0, len(l)) # random int ranging from 0 to len()-1
	random_choice = l[random_index]
	l.pop(random_index)
	return random_choice
#####################

### Classes ###
class Cell_Gradient:
	"""
	Container class for dC_t/d_W of each cell.
	"""
	def __init__(self, inp_size: int, out_size: int):
		# gates
		# C is independent of the Output gate
		self.d_w_xi = np.zeros([out_size, out_size, inp_size])
		self.d_w_hi = np.zeros([out_size, out_size, out_size])
		self.d_w_ci = np.zeros([out_size, out_size, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_i = np.zeros([out_size, out_size, 1])

		self.d_w_xf = np.zeros([out_size, out_size, inp_size])
		self.d_w_hf = np.zeros([out_size, out_size, out_size])
		self.d_w_cf = np.zeros([out_size, out_size, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_f = np.zeros([out_size, out_size, 1])
		
		# external weights
		self.d_w_xc = np.zeros([out_size, out_size, inp_size])
		self.d_w_hc = np.zeros([out_size, out_size, out_size])
		self.d_b_c = np.zeros([out_size, out_size, 1])
		
	def reset(self):
		self.d_w_xi.fill(0.);	self.d_w_hi.fill(0.)
		self.d_w_ci.fill(0.);	self.d_b_i.fill(0.)

		self.d_w_xf.fill(0.);	self.d_w_hf.fill(0.)
		self.d_w_cf.fill(0.);	self.d_b_f.fill(0.)

		self.d_w_xc.fill(0.);	self.d_w_hc.fill(0.)
		self.d_b_c.fill(0.)

class Gradient(object):	
	"""
	Container class for dE_t/d_W for each W, aka the total gradient at timestamp t. 
	"""
	def __init__(self, inp_size: int, out_size: int):
		# gates
		self.d_w_xi = np.zeros([out_size, inp_size]);	self.d_w_hi = np.zeros([out_size, out_size])
		self.d_w_ci = np.zeros([out_size, 1]);	self.d_b_i = np.zeros([out_size, 1])

		self.d_w_xf = np.zeros([out_size, inp_size]);	self.d_w_hf = np.zeros([out_size, out_size])
		self.d_w_cf = np.zeros([out_size, 1]);	self.d_b_f = np.zeros([out_size, 1])

		self.d_w_xo = np.zeros([out_size, inp_size]);	self.d_w_ho = np.zeros([out_size, out_size])
		self.d_w_co = np.zeros([out_size, 1]);	self.d_b_o = np.zeros([out_size, 1])
		
		self.d_w_xc = np.zeros([out_size, inp_size]);	self.d_w_hc = np.zeros([out_size, out_size])
		self.d_b_c = np.zeros([out_size, 1])
		
		# external weights
		self.d_w_hy = np.zeros([out_size, out_size]);	self.d_b_y = np.zeros([out_size, 1])
		
	def incr(self, other):
		# += operator
		self.d_w_xi += other.d_w_xi;	self.d_w_hi += other.d_w_hi
		self.d_w_ci += other.d_w_ci;	self.d_b_i += other.d_b_i
		
		self.d_w_xf += other.d_w_xf;	self.d_w_hf += other.d_w_hf
		self.d_w_cf += other.d_w_cf;	self.d_b_f += other.d_b_f
		
		self.d_w_xo += other.d_w_xo;	self.d_w_ho += other.d_w_ho
		self.d_w_co += other.d_w_co;	self.d_b_o += other.d_b_o

		self.d_w_xc += other.d_w_xc;	self.d_w_hc += other.d_w_hc
		self.d_b_c += other.d_b_c
		
		self.d_w_hy += other.d_w_hy;	self.d_b_y += other.d_b_y
		
	def div(self, factor):
		# /= operator
		self.d_w_xi /= factor;	self.d_w_hi /= factor
		self.d_w_ci /= factor;	self.d_b_i /= factor

		self.d_w_xf /= factor;	self.d_w_hf /= factor
		self.d_w_cf /= factor;	self.d_b_f /= factor

		self.d_w_xo /= factor;	self.d_w_ho /= factor
		self.d_w_co /= factor;	self.d_b_o /= factor

		self.d_w_xc /= factor;	self.d_w_hc /= factor
		self.d_b_c /= factor

		self.d_w_hy /= factor;	self.d_b_y /= factor
###########

