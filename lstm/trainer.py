from lstm import *

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
		self.d_w_xi.fill(0.)
		self.d_w_hi.fill(0.)
		self.d_w_ci.fill(0.)
		self.d_b_i.fill(0.)

		self.d_w_xf.fill(0.)
		self.d_w_hf.fill(0.)
		self.d_w_cf.fill(0.)
		self.d_b_f.fill(0.)

		self.d_w_xc.fill(0.)
		self.d_w_hc.fill(0.)
		self.d_b_c.fill(0.)

class Gradient(object):	
	"""
	Container class for dE_t/d_W for each W, aka the total gradient at timestamp t. 
	"""
	def __init__(self, inp_size: int, out_size: int):
		# gates
		self.d_w_xi = np.zeros([out_size, inp_size]);	self.d_w_hi = np.zeros([out_size, inp_size])
		self.d_w_ci = np.zeros([out_size, 1]);	self.d_b_i = np.zeros([out_size, 1])

		self.d_w_xf = np.zeros([out_size, inp_size]);	self.d_w_hf = np.zeros([out_size, inp_size])
		self.d_w_cf = np.zeros([out_size, 1]);	self.d_b_f = np.zeros([out_size, 1])

		self.d_w_xo = np.zeros([out_size, inp_size]);	self.d_w_ho = np.zeros([out_size, inp_size])
		self.d_w_co = np.zeros([out_size, 1]);	self.d_b_o = np.zeros([out_size, 1])
		
		self.d_w_xc = np.zeros([out_size, inp_size]);	self.d_w_hc = np.zeros([out_size, inp_size])
		self.d_b_c = np.zeros([out_size, 1])
		
		# external weights
		self.d_w_hy = np.zeros([out_size, inp_size]);	self.d_b_y = np.zeros([out_size, 1])
		
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

class LSTM_Trainer:
	def __init__(self, lstm : LSTM):
		self.lstm = lstm
		self.cell_grad = Cell_Gradient(lstm.inp_size, lstm.out_size)
		#self.gradient = Gradient(lstm.inp_size)
	
	def forward_backward_prop(self, x_t, l_t, cost_function:str = 'MSE'):
		'''
		Calculates the total gradient and updates weights 
		'''
		inp = self.lstm.inp_size
		out = self.lstm.out_size
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
		self.cell_grad.d_w_xi = vec_dot_ten(z_t*aux_di, vec2ten(x_t, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_xi) 
		self.cell_grad.d_w_hi = vec_dot_ten(z_t*aux_di, vec2ten(h_t_, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_hi) 
		self.cell_grad.d_w_ci = vec_dot_ten(z_t*aux_di, vec2diag_mat(c_t_)) + vec_dot_ten(f_t, self.cell_grad.d_w_ci)
		self.cell_grad.d_b_i = vec_dot_ten(z_t*aux_di, vec2diag_mat(col(np.ones(out)))) + vec_dot_ten(f_t, self.cell_grad.d_b_i)
		
		aux_df = d_sigmoid(in_f)
		self.cell_grad.d_w_xf = vec_dot_ten(c_t_*aux_df, vec2ten(x_t, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_xf)
		self.cell_grad.d_w_hf = vec_dot_ten(c_t_*aux_df, vec2ten(h_t_, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_hf)
		self.cell_grad.d_w_cf = vec_dot_ten(c_t_*aux_df, vec2diag_mat(c_t_)) + vec_dot_ten(f_t, self.cell_grad.d_w_cf)
		self.cell_grad.d_b_f = vec_dot_ten(c_t_*aux_df, vec2diag_mat(col(np.ones(out)))) + vec_dot_ten(f_t, self.cell_grad.d_b_f)
		
		aux_dz = d_sigmoid(in_z)
		self.cell_grad.d_w_xc = vec_dot_ten(i_t*aux_dz, vec2ten(x_t, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_xc)
		self.cell_grad.d_w_hc = vec_dot_ten(i_t*aux_dz, vec2ten(h_t_, out)) + vec_dot_ten(f_t, self.cell_grad.d_w_hc)
		self.cell_grad.d_b_c = vec_dot_ten(i_t*aux_dz, vec2diag_mat(col(np.ones(out)))) + vec_dot_ten(f_t, self.cell_grad.d_b_c)
		###
		
		if cost_function == 'MSE':
			# delta_h = row vector
			dE_dy = y_t-l_t
			delta_h = np.dot( row(dE_dy), self.lstm.d_Y(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y) * self.lstm.w_hy )
		aux = o_t * d_tanh(c_t)
		
		gradient = Gradient(inp, out)
		
		### input gate
		gradient.d_w_xi = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_xi), axes=1)
		gradient.d_w_hi = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_hi), axes=1)
		gradient.d_w_ci = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_ci), axes=1)
		gradient.d_b_i = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_b_i), axes=1)
		###
		
		### forget gate
		gradient.d_w_xf = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_xf), axes=1)
		gradient.d_w_hf = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_hf), axes=1)
		gradient.d_w_cf = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_cf), axes=1)
		gradient.d_b_f = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_b_f), axes=1)
		###
		
		### cell gate
		gradient.d_w_xc = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_xc), axes=1)
		gradient.d_w_hc = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_w_hc), axes=1)
		gradient.d_b_c = np.tensordot(delta_h, vec_dot_ten(aux, self.cell_grad.d_b_c), axes=1)
		###
		
		### output gate
		#aux_o = delta_h * tanh(c_t) * d_sigmoid(in_o) # acho que é col(delta_h) * ......
		aux_o = col(delta_h) * tanh(c_t) * d_sigmoid(in_o)
		gradient.d_w_xo = aux_o * vec2full_mat(x_t, out)
		gradient.d_w_ho = aux_o * vec2full_mat(h_t_, out)
		gradient.d_w_co = aux_o * c_t_
		gradient.d_b_o = aux_o
		###
		
		### lstm output weights
		aux_y = dE_dy * self.lstm.d_Y(np.dot(self.lstm.w_hy, h_t) + self.lstm.b_y)
		gradient.d_w_hy = aux_y * vec2full_mat(h_t, out)
		gradient.d_b_y = aux_y
		###
		
		return gradient 
	
	def update_weights(self, learning_rate, grad : Gradient):
		self.lstm.block.w_xi += -learning_rate * grad.d_w_xi 
		self.lstm.block.w_hi += -learning_rate * grad.d_w_hi 
		self.lstm.block.w_ci += -learning_rate * grad.d_w_ci 
		self.lstm.block.b_i += -learning_rate * grad.d_b_i 
		
		self.lstm.block.w_xf += -learning_rate * grad.d_w_xf 
		self.lstm.block.w_hf += -learning_rate * grad.d_w_hf 
		self.lstm.block.w_cf += -learning_rate * grad.d_w_cf 
		self.lstm.block.b_f += -learning_rate * grad.d_b_f 
		
		self.lstm.block.w_xo += -learning_rate * grad.d_w_xo 
		self.lstm.block.w_ho += -learning_rate * grad.d_w_ho 
		self.lstm.block.w_co += -learning_rate * grad.d_w_co 
		self.lstm.block.b_o += -learning_rate * grad.d_b_o 
		
		self.lstm.block.w_xc += -learning_rate * grad.d_w_xc 
		self.lstm.block.w_hc += -learning_rate * grad.d_w_hc 
		self.lstm.block.b_c += -learning_rate * grad.d_b_c 
		
		self.lstm.w_hy += -learning_rate * grad.d_w_hy
		self.lstm.b_y += -learning_rate * grad.d_b_y

	def train(self, x_inputs:list, labels: list, learning_rate, batch_size:int, n_epochs:int):
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
				batch_gradient = Gradient(self.lstm.inp_size, self.lstm.out_size) # total gradient of this batch
				self.cell_grad.reset()
				self.lstm.reset_block()
				self.lstm.initialize()
				t0 = select_and_pop(list_starting_points)
				
				for i in range(batch_size):
					grad = self.forward_backward_prop(x_inputs[t0+i], labels[t0+i])
					grad.div(batch_size)
					print(grad.d_w_xi.shape)
					print(batch_gradient.d_w_xi.shape)
					batch_gradient.incr(grad)
				
				self.update_weights(learning_rate, batch_gradient)