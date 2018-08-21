from trainer import *

T = 3
input_length = 1000
lstm = LSTM(T, input_length)

# create list of inputs
inputs = []
for i in range(T):
	inputs.append(np.random.randn(input_length, 1))
	
# forward_pass test
out = lstm.forward_pass(inputs) # todos outputs tao iguais, mudar isso
for o in out:
	print(o)
	
# trainer test
trainer = LSTM_Trainer(lstm)
trainer.forward_backward_prop(inputs[0], inputs[1])