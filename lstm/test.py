from lstm import *

def main():

	
	# lstm_cell test
	dim = 10000
	ls = LSTM_Cell(dim)
	x_t = np.random.randn(dim, 1)
	h_t_ = np.random.randn(dim, 1)
	c_t_ = np.random.randn(dim, 1)

	h_t, c_t = ls.compute(x_t, h_t_, c_t_)

	print(h_t.shape, c_t.shape)
	print(h_t[:5], c_t[:5])
	

if __name__ == "__main__":
	main()
