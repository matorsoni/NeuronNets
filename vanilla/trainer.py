from neuron_network import *

class Trainer:

    def __init__(self, nn:NeuronNetwork, x_train, label_train, x_test, label_test):
        self.nn=nn
        self.x_train=x_train # np.array column
        self.label_train=label_train
        self.x_test=x_test # np.array column
        self.label_test=label_test

    '''
    Gradient and backpropagation algorithm. Variable names follow the notation from this
    reference: http://neuralnetworksanddeeplearning.com/chap2.html
    Note: the index l goes from 0 to L-1 instead of 1 to L.
    '''
    def gradient(self, x_input, cost_function='MSE'):
        # Gradient of the cost function with respect to the weights and biases
        # for a given input x_input.
        # C = 1/2 (y(x_input) - y_label)²
        L = len(self.nn.dimensions)
        delta=[]
        z=[]

        




        pass

    def train(batch_size=100, epochs=1, learn_rate=0.001):

        for m in range(batch_size):
            pass


c=np.array([[1.], [2.], [3.]])
print(c.size)
