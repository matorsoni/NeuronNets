from lstm_simple import *

T = 3
input_length = 10
lstm = LSTM_Simple(T, input_length)

inputs = []
for i in range(T):
	inputs.append(np.random.randn(input_length, 1))
	
out = lstm.forward_pass(inputs) # todos outputs tao iguais, mudar isso
for o in out:
	print(o)