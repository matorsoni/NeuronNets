import numpy as np

class functions:

	def sigmoid(z):
		return 1./(1.+np.exp(-z))

	def d_sigmoid(z):
		return functions.sigmoid(z)*(1.-functions.sigmoid(z))

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
			return functions.sigmoid, functions.d_sigmoid
		if func_name == 'relu':
			return functions.relu, functions.d_relu