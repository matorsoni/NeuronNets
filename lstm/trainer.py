from lstm import *

class d_Cell:
	"""
	Container class for dC_t/d_W of each cell.
	"""
	def __init__(self, input_length: int):
		# gates
		# C is independent of the Output gate
		self.d_w_xi = np.zeros([input_length, input_length, input_length])
		self.d_w_hi = np.zeros([input_length, input_length, input_length])
		self.d_w_ci = np.zeros([input_length, input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_i = np.zeros([input_length, input_length, 1])

		self.d_w_xf = np.zeros([input_length, input_length, input_length])
		self.d_w_hf = np.zeros([input_length, input_length, input_length])
		self.d_w_cf = np.zeros([input_length, input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_f = np.zeros([input_length, input_length, 1])
		
		# external weights
		self.d_w_xc = np.zeros([input_length, input_length, input_length])
		self.d_w_hc = np.zeros([input_length, input_length, input_length])
		self.d_b_c = np.zeros([input_length, input_length, 1])

class Gradient:
	"""
	Container class for dE_t/d_W for each W, aka the total gradient at timestamp t. 
	"""
	def __init__(self, input_length: int):
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
		self.d_w_hy = np.zeros([input_length, input_length])
		self.d_b_y = np.zeros([input_length, 1])

class LSTM_Trainer:
	def __init__(self, lstm : LSTM):
		self.lstm = lstm
		self.d_cell = d_Cell(lstm.input_length)
		self.gradient = Gradient(lstm.input_length)
		# colocar as d_w da lstm 
	
	def forward_backward_prop(self, x_t, l_t, cost_function:str = 'MSE'):
		'''
		Calculates the total gradient and updates gradient 
		'''
		# forward prop:
		in_i, in_f, in_o, in_z, c_t, h_t = self.lstm.block.compute(x_t, True)
		i_t = sigmoid(in_i)
		f_t = sigmoid(in_f)
		z_t = sigmoid(in_z)
		o_t = sigmoid(in_o)
		c_t_ = self.lstm.block.get_c(0)
		h_t_ = self.lstm.block.get_h(0)
		y_t = sigmoid(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y)
		
		# backprop:
		
		### gradient of the cell state c_t
		aux_di = d_sigmoid(in_i)
		self.d_cell.d_w_xi = vec_dot_ten(z_t*aux_di, vec2ten(x_t)) + vec_dot_ten(f_t, self.d_cell.d_w_xi) 
		self.d_cell.d_w_hi = vec_dot_ten(z_t*aux_di, vec2ten(h_t_)) + vec_dot_ten(f_t, self.d_cell.d_w_hi) 
		self.d_cell.d_w_ci = vec_dot_ten(z_t*aux_di, vec2diag_mat(c_t_)) + vec_dot_ten(f_t, self.d_cell.d_w_ci)
		self.d_cell.d_b_i = vec_dot_ten(z_t*aux_di, vec2diag_mat(col(np.ones(x_t.size)))) + vec_dot_ten(f_t, self.d_cell.d_b_i)
		
		aux_df = d_sigmoid(in_f)
		self.d_cell.d_w_xf = vec_dot_ten(c_t_*aux_df, vec2ten(x_t)) + vec_dot_ten(f_t, self.d_cell.d_w_xf)
		self.d_cell.d_w_hf = vec_dot_ten(c_t_*aux_df, vec2ten(h_t_)) + vec_dot_ten(f_t, self.d_cell.d_w_hf)
		self.d_cell.d_w_cf = vec_dot_ten(c_t_*aux_df, vec2diag_mat(c_t_)) + vec_dot_ten(f_t, self.d_cell.d_w_cf)
		self.d_cell.d_b_f = vec_dot_ten(c_t_*aux_df, vec2diag_mat(col(np.ones(x_t.size)))) + vec_dot_ten(f_t, self.d_cell.d_b_f)
		
		aux_dz = d_sigmoid(in_z)
		self.d_cell.d_w_xc = vec_dot_ten(i_t*aux_dz, vec2ten(x_t)) + vec_dot_ten(f_t, self.d_cell.d_w_xc)
		self.d_cell.d_w_hc = vec_dot_ten(i_t*aux_dz, vec2ten(h_t_)) + vec_dot_ten(f_t, self.d_cell.d_w_hc)
		self.d_cell.d_b_c = vec_dot_ten(i_t*aux_dz, vec2diag_mat(col(np.ones(x_t.size)))) + vec_dot_ten(f_t, self.d_cell.d_b_c)
		###
		
		if cost_function == 'MSE':
			# delta_h = row vector
			dE_dy = y_t-l_t
			delta_h = np.dot( row(dE_dy), self.lstm.d_Y(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y) * self.lstm.w_hy )
		aux = o_t * d_tanh(c_t)
		
		### input gate
		self.gradient.d_w_xi = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_xi), axes=1)
		self.gradient.d_w_hi = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_hi), axes=1)
		self.gradient.d_w_ci = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_ci), axes=1)
		self.gradient.d_b_i = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_b_i), axes=1)
		###
		
		### forget gate
		self.gradient.d_w_xf = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_xf), axes=1)
		self.gradient.d_w_hf = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_hf), axes=1)
		self.gradient.d_w_cf = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_cf), axes=1)
		self.gradient.d_b_f = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_b_f), axes=1)
		###
		
		### cell gate
		self.gradient.d_w_xc = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_xc), axes=1)
		self.gradient.d_w_hc = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_w_hc), axes=1)
		self.gradient.d_b_c = np.tensordot(delta_h, vec_dot_ten(aux, self.d_cell.d_b_c), axes=1)
		###
		
		### output gate
		#aux_o = delta_h * tanh(c_t) * d_sigmoid(in_o) # acho que é col(delta_h) * ......
		aux_o = col(delta_h) * tanh(c_t) * d_sigmoid(in_o)
		self.gradient.d_w_xo = aux_o * vec2full_mat(x_t)
		self.gradient.d_w_ho = aux_o * vec2full_mat(h_t_)
		self.gradient.d_w_co = aux_o * c_t_
		self.gradient.d_b_o = aux_o
		###
		
		### lstm output weights
		aux_y = dE_dy * self.lstm.d_Y(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y)
		self.gradient.d_w_hy = aux_y * vec2full_mat(h_t)
		self.gradient.d_b_y = aux_y
		###
	
	def update_weights(self, learning_rate):
		self.lstm.block.w_xi += learning_rate * self.gradient.d_w_xi 
		self.lstm.block.w_hi += learning_rate * self.gradient.d_w_hi 
		self.lstm.block.w_ci += learning_rate * self.gradient.d_w_ci 
		self.lstm.block.b_i += learning_rate * self.gradient.d_b_i 
		
		self.lstm.block.w_xf += learning_rate * self.gradient.d_w_xf 
		self.lstm.block.w_hf += learning_rate * self.gradient.d_w_hf 
		self.lstm.block.w_cf += learning_rate * self.gradient.d_w_cf 
		self.lstm.block.b_f += learning_rate * self.gradient.d_b_f 
		
		self.lstm.block.w_xo += learning_rate * self.gradient.d_w_xo 
		self.lstm.block.w_ho += learning_rate * self.gradient.d_w_ho 
		self.lstm.block.w_co += learning_rate * self.gradient.d_w_co 
		self.lstm.block.b_o += learning_rate * self.gradient.d_b_o 
		
		self.lstm.block.w_xc += learning_rate * self.gradient.d_w_xc 
		self.lstm.block.w_hc += learning_rate * self.gradient.d_w_hc 
		self.lstm.block.b_c += learning_rate * self.gradient.d_b_c 
		
		self.lstm.w_hy += learning_rate * self.gradient.d_w_hy
		self.lstm.b_y += learning_rate * self.gradient.d_b_y

	def train(self, x_inputs:list, learning_rate, batch_size:int, n_epochs:int):
		'''
		
		'''
		n_examples = len(x_inputs)
		assert batch_size <= n_examples
		# every batch runs through batch_size+1 inputs (we take x[t] as input and x[t+1] as label)
		# so the possible starting points are n_examples - batch_size + 1 - 1 
		n_starting_points = n_examples - batch_size
		
		for epoch in range(n_epochs):
			print( ('Epoch '+'{} / {}').format(epoch+1, n_epochs) )
			list_starting_points = list(range(n_starting_points))
			
			# randomly picks all possible starting points
			while len(list_starting_points) > 0:
				print(len(list_starting_points))
				self.lstm.initialize()
				t0 = select_and_pop(list_starting_points)
				for i in range(batch_size):
					self.forward_backward_prop(x_inputs[t0+i], x_inputs[t0+i+1])
					self.update_weights(learning_rate)
				
				
			
			
			
		
		
		
		
		
		
		
		
		