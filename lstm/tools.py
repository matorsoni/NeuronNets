import numpy as np
from copy import deepcopy


### functions
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

### matrix algebra
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
	
### miscellaneous
def select_and_pop(l:list):
	# chooses a random value in the list and pop it
	random_index = np.random.randint(0, len(l)) # random int ranging from 0 to len()-1
	random_choice = l[random_index]
	l.pop(random_index)
	return random_choice
	


