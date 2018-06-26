import numpy as np
from mnist import MNIST

class functions:

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def d_sigmoid(z):
        return sigmoid(z)*(1-sigmoid(z))

    def relu(z):
        return 0 if z<=0. else z

    def d_relu(z):
        return 0 if z<=0. else 1

    def choose(func_name = 'sigmoid'):
        if func_name == 'sigmoid':
            return functions.sigmoid, functions.d_sigmoid
        if func_name == 'relu':
            return functions.relu, functions.d_relu

class NeuronNetwork:

    def __init__(self, input_size, output_size, hidden_layers=[], activation='sigmoid'):
        self.dimensions = [input_size] + hidden_layers + [output_size]
        self.act, self.d_act = functions.choose(activation)

        # random initialization of weights and biases
        # format: w[n_layer][w_ji (matrix)] , b[n_layer][b_j (column vec)]
        w_list = []
        b_list = []
        for l in range(len(hidden_layers) + 1):
            w_list.append(np.random.randn(self.dimensions[l+1], self.dimensions[l]))
            b_list.append(
                np.random.randn(self.dimensions[l+1]).reshape((self.dimensions[l+1],1))
                )

        self.w = np.array(w_list)
        self.b = np.array(b_list)

    def predict(self, x):
        pass

    def train(self, batch_size=10, epoch=1):
        pass


#nn = NeuronNetwork(input_size=784, output_size=10, hidden_layers=[15])
#print(nn.b[0])
#print(nn.b[0].reshape(nn.b[0].size,1))
#print(np.array([1,1,1,1,1]))


# teste

#import data
mndata = MNIST('../data/')
images, labels = mndata.load_training()
train = np.array(images) # train[0 - 59999][0 - 783]
labels = np.array(labels) # labels[0 - 59999]
for n in labels[0:20]:
    print(n)
