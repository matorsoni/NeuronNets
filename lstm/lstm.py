from lstm_block import *

class LSTM:
	def __init__(self, input_length: int, output_length: int, y_activation = 'sigmoid'):
		'''
		Simplest version of a LSTM (with peepholes), see figures at https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1
		T = total number of timestamps 
		'''
		self.input_length = input_length
		self.output_length = output_length
		self.block = LSTM_Block(1, input_length, output_length)
		self.Y, self.d_Y = choose(y_activation)
		
		## Se pá nem precisa dessas paradas... os gates já tem weighted inputs no lstm_block
		#self.W_ih = np.random.randn(input_length, input_length) # weighted input x_t
		#self.W_hh = np.random.randn(input_length, input_length) # weighted recurrent input h_t_
		#self.b_h = np.random.randn(input_length, 1) # biased recurrent input
		self.w_hy = np.random.randn(output_length, output_length) # weighted output y_t
		self.b_y = np.random.randn(output_length, 1) # biased output y_t
	
	def initialize(self):
		self.block.compute(np.zeros([self.input_length, 1]))
		
	def single_forward_pass(self, x_t):
		self.block.compute(x_t)
		y = self.Y(self.b_y + np.dot(self.w_hy, self.block.get_h(-1))) # Y(y), Y=sigmoid -> can be changed
		return y
	def forward_pass(self, x_inputs:list):
		# input = list[x_0, x_1, x_2, ...], x_i = column np.array
		output_list = []
		for t in range(len(x_inputs)):
			y = self.single_forward_pass(x_inputs[t])
			output_list.append(deepcopy(y))  
		
		return output_list
		
	#def save_model(self):
		#pickle