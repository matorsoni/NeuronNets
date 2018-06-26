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
    Note: the index l goes from 0 to L-2 instead of 2 to L.
    '''
    def gradient(self, x_input, label_train, cost_function='MSE'):
        '''
        Gradient of the cost function with respect to the weights and biases
        for a given input x_input.
        C = 1/2 (y(x_input) - y_label)²
        '''
        assert x_input.shape==(self.nn.dimensions[0],), "Incompatible input format"
        L = len(self.nn.dimensions)

        # one-hot column vector enconding of label_train:
        y_label = np.zeros(self.nn.dimensions[L-1]).reshape(self.nn.dimensions[L-1],1)
        y_label[label_train][0]=1.

        # Forward propagation to compute the weighted sums z[l][j]
        y_prediction = x_input.reshape(self.nn.dimensions[0],1) # turns input into column vector
        z=[]
        for l in range(L-1):
            z.append( np.dot( self.nn.w[l], y_prediction ) + self.nn.b[l] )
            y_prediction = self.nn.act( z[l] )
        z = np.array(z)

        # Backward propagation to compute the derivatives of C wrt z[l][j],
        delta=[]
        delta.append( (y_prediction-y_label)*self.nn.d_act(z[L-2]) ) #element-wise product
        if L>2:
            for l in range(L-3, -1):
                # Backward recurrence: delta[l]=sigma'(z[l]) * np.dot(w[l+1].transpose, delta[l+1])
                # delta[0] is always the most recent !
                delta = [self.nn.d_act(z[l])*np.dot( self.nn.w[l+1].transpose(), delta[0] )]+delta
        delta = np.array(delta)

        # Finally, compute the actual gradients in function of z[] and delta[]
        grad_cost_w = []
        for l in range(L-1)
            if l==0:
                grad_cost_w.append( delta[l] * x_input) #column vec*row vec = tensor product = matrix
            else:
                #delta (column vec) * sigma(z[l-1]) (row vec)
                grad_cost_w.append( delta[l] * self.nn.act(z[l-1].reshape(z[l-1].size)) )

        grad_cost_w = np.array(grad_cost_w)
        grad_cost_b = delta
        return grad_cost_w, grad_cost_b

    def train(batch_size=100, epochs=1, learn_rate=0.001):

        for m in range(batch_size):
            pass
