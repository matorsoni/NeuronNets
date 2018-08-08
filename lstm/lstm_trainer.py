from lstm import *

class d_Cell:
	"""
	Container class for the d_W's of each cell.
	"""
	def __init__(self, input length: int):
		self.d_w_xi = np.zeros([input_length, input_length])
		self.d_w_hi = np.zeros([input_length, input_length])
		self.d_w_ci = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_i = np.zeros([input_length, 1])

		self.d_w_xf = np.zeros([input_length, input_length])
		self.d_w_hf = np.zeros([input_length, input_length])
		self.d_w_cf = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_f = np.zeros([input_length, 1])

		self.d_w_xc = np.zeros([input_length, input_length])
		self.d_w_hc = np.zeros([input_length, input_length])
		self.d_b_c = np.zeros([input_length, 1])

		self.d_w_xo = np.zeros([input_length, input_length])
		self.d_w_ho = np.zeros([input_length, input_length])
		self.d_w_co = np.zeros([input_length, 1]) # diagonal matrix, so make it into a column vector
		self.d_b_o = np.zeros([input_length, 1])
		
class LSTM_Trainer:
	def __init__(self, lstm, cost_func, ):
		self.lstm = lstm
		'''
		criar variaveis d_W_... pra cada matriz W e b da lstm
		criar uma lista trainer_cells contendo d_w... de cada matriz das cells 
		setar todas as variaveis d_... = 0 no init
		
		'''
		
		self.d_W_ih = [np.random.randn(lstm.input_length, lstm.input_length) for i in range(lstm.n_layers)] # weights from input to cells n
		self.d_W_hh = [np.random.randn(lstm.input_length, lstm.input_length) for i in range(lstm.n_layers)] # weights from layers t-1 to t, cells n to n
		self.d_W_h_h = [np.random.randn(lstm.input_length, lstm.input_length) for i in range(lstm.n_layers-1)] # weights from layer t to t, cell n-1 to n
		self.d_b_h = [np.random.randn(lstm.input_length, 1) for i in range(lstm.n_layers)] # biases in each cell in layer t
		self.d_W_hy = [np.random.randn(lstm.input_length, lstm.input_length) for i in range(lstm.n_layers)] # weights from cells n to output y
		self.d_b_y = np.random.randn(lstm.input_length, 1)
		
		self.d_cells = [ d_Cell(lstm.input_length) for n in range(lstm.n_layers) ] # d_cells[i] = Set of d_W's corresponding to i-th cell.
		
	def update_cell(self, d_cell : d_Cell):
		pass
	
	def train_single_batch(self, x_inputs, y_labels, learn_rate, 