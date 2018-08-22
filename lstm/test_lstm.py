from trainer import *


n_training_examples = 100
input_length = 100
lstm = LSTM(input_length)

# create list of inputs
x_inputs = []
for i in range(n_training_examples):
	v = np.zeros([input_length, 1])
	v[i][0] = 1
	x_inputs.append(deepcopy(v))
	
# forward_pass test
out = lstm.forward_pass(x_inputs) 
	
# trainer test
trainer = LSTM_Trainer(lstm)
trainer.forward_backward_prop(x_inputs[0], x_inputs[1])

trainer.train(x_inputs, learning_rate=0.001, batch_size=20, n_epochs=10)