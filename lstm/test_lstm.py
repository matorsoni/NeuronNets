from trainer import *


n_training_examples = 10
inp_size = 10
out_size = 8
lstm = LSTM(inp_size, out_size)

# create lists of inputs and labels
x_inputs = []
labels = []
map = np.random.randn(out_size, inp_size)
for i in range(n_training_examples):
	v = np.zeros([inp_size, 1])
	v[i][0] = 1
	x_inputs.append(deepcopy(v))
	
	labels.append(deepcopy(np.dot(map, v))) 
	
# forward_pass test
out = lstm.forward_pass(x_inputs) 
	
# trainer test
trainer = LSTM_Trainer(lstm)
trainer.forward_backward_prop(x_inputs[0], labels[0])

trainer.train(x_inputs, labels, learning_rate=0.0001, batch_size=5, n_epochs=1000)
print(x_inputs[0])
print(lstm.single_forward_pass(x_inputs[0]))
print(labels[0])