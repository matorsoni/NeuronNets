import numpy as np


# functions
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

# matrix algebra
def col(v):
	return v.reshape(v.size,1)
	
def row(v):
	return v.reshape(v.size)

def vec2mat(v):
	# constructs a matrix whose rows are v
	if v.shape == (v.size,1):
		v = row(v)
	return np.array([v for i in range(v.size)])

