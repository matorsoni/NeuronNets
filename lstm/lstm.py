from lstm_block import *

class LSTM:
	def __init__(self, T: int, input_length: int, y_activation = 'sigmoid'):
		'''
		Simplest version of a LSTM (with peepholes), see figures at https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1
		'''
		self.T = T
		self.input_length = input_length
		self.block = LSTM_Block(1, input_length)
		self.Y, self.d_Y = choose(y_activation)
		
		## Se pá nem precisa dessas paradas... os gates já tem weighted inputs no lstm_block
		#self.W_ih = np.random.randn(input_length, input_length) # weighted input x_t
		#self.W_hh = np.random.randn(input_length, input_length) # weighted recurrent input h_t_
		#self.b_h = np.random.randn(input_length, 1) # biased recurrent input
		self.W_hy = np.random.randn(input_length, input_length) # weighted output y_t
		self.b_y = np.random.randn(input_length, 1) # biased output y_t
	
	def forward_pass(self, x_inputs:list):
		# input = list[x_0, x_1, x_2, ...], x_i = column np.array
		assert len(x_inputs) == self.T, 'Incompatible number of x_inputs'
		output_list = []
		for t in range(self.T):
			'''
			if t==0: # Acho que o primeiro pass deve ser com x = zeros()
				self.block.compute(np.dot(self.W_ih, x_inputs[t]) + self.b_h)
			else:
				self.block.compute(
					np.dot(self.W_ih, x_inputs[t]) + np.dot(self.W_hh, self.block.get_h(-1)) + 
					self.b_h	)
			'''
			self.block.compute(x_inputs[t])
			y = self.Y(self.b_y + np.dot(self.W_hy, self.block.get_h(-1))) # Y(y), Y=sigmoid -> can be changed
			
			output_list.append(deepcopy(y))  
		
		return output_list
		
	#def save_model(self):
		#pickle