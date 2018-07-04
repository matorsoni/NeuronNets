from neuron_network import *
from copy import deepcopy


class Trainer:

    def __init__(self, nn:NeuronNetwork):
        self.nn=nn

    def one_hot_vector(self, label_input:int):
        '''
        One-hot column vector enconding of label_input
        '''
        vec_label = np.zeros(self.nn.layers[-1]).reshape(self.nn.layers[-1],1)
        vec_label[label_input][0]=1.
        return vec_label

    '''
    Gradient and backpropagation algorithm. Variable names follow the notation from this
    reference: http://neuralnetworksanddeeplearning.com/chap2.html
    Note: the index l goes from 0 to L-2 instead of 2 to L.
    '''
    def gradient(self, x_input, label_input:int, cost_function='MSE'):
        '''
        Gradient of the cost function with respect to the weights and biases
        for a given input x_input.
        C = 1/2 (y(x_input) - y_label)²

        input format:
            x_input = column np.array, shape = (___, 1)
            label_input = integer 0,...,9

        variables in this scope:
            delta = np.array, with delta[l] = column vector
            z = np.array, with z[l] = column vector

        output format:
            grad_cost_w = np.array, with grad_cost_w[l] = matrix
            grad_cost_b = np.array, with grad_cost_b[l] = column vector

        '''

        assert x_input.shape == (self.nn.layers[0], 1) , "Incompatible input dimension"
        L = len(self.nn.layers)

        y_label = self.one_hot_vector(label_input)

        # Forward propagation to compute the weighted sums z[l][j]
        y_prediction = np.array(x_input, dtype=np.float64) # trying to avoid numerical error
        z=[]
        for l in range(L-1):
            z.append( np.dot( self.nn.w[l], y_prediction ) + self.nn.b[l] )
            y_prediction = self.nn.act( z[l] )
        z = np.array(z)

        # Backward propagation to compute the derivatives of C wrt z[l][j],
        delta=[]
        delta.append( (y_prediction-y_label)*self.nn.d_act(z[L-2]) ) #element-wise product

        if L>2:
            for l in range(L-3, -1, -1): # from L-3 to 0 with step=-1
                # Backward recurrence: delta[l]=sigma'(z[l]) * np.dot(w[l+1].transpose, delta[l+1])
                # delta[0] is always the most recent !
                delta = [ self.nn.d_act(z[l])*np.dot( self.nn.w[l+1].transpose(), delta[0] ) ]+delta
        delta = np.array(delta)

        # Finally, compute the actual gradients in function of z[] and delta[]

        grad_cost_w = [ delta[0] * x_input.reshape(x_input.size) ] #column vec*row vec = tensor product = matrix
        if L>2:
            for l in range(1, L-1):
            #if l==0:
            #    grad_cost_w.append( delta[l] * x_input) #column vec*row vec = tensor product = matrix
            #else:
                #delta (column vec) * sigma(z[l-1]) (row vec)
                grad_cost_w.append( delta[l] * self.nn.act(z[l-1].reshape(z[l-1].size)) )

        grad_cost_w = np.array(grad_cost_w)
        grad_cost_b = delta # useless, might as well just return delta....
        return grad_cost_w, grad_cost_b

    def train(self, x_train, labels_train, n_training_examples:int,
        batch_size=100, n_epochs=1, learn_rate=0.001):
        '''
        Training method.

        input format:
            x_train[n_example] = np.array column vector
            label_train[n_example] = 0,...,9
        '''
        batches_per_epoch = int(n_training_examples/batch_size)
        for epoch in range(n_epochs):
            print( ('Epoch '+'{} / {}').format(epoch+1, n_epochs) )

            for batch in range(batches_per_epoch):
                #print( ('Batch '+'{} / {}').format(batch+1, batches_per_epoch) )
                # Creates an empty np.array with the same dimensions as nn.w and nn.b
                grad_cost_w_total = deepcopy(self.nn.w)
                grad_cost_w_total.fill(0.)
                grad_cost_b_total = deepcopy(self.nn.b)
                grad_cost_b_total.fill(0.)

                # calculates as averages the gradients for one batch
                for n_example in range(batch_size):
                    x_input = x_train[batch*batch_size+n_example]
                    label_input = labels_train[batch*batch_size+n_example]
                    grad_cost_w, grad_cost_b = self.gradient(x_input, label_input)

                    grad_cost_w_total += grad_cost_w # may cause numerical errors??
                    grad_cost_b_total += grad_cost_b

                grad_cost_w_total = grad_cost_w_total/batch_size # numerical errors??
                grad_cost_b_total = grad_cost_b_total/batch_size

                #updates the weigths:
                self.nn.w = self.nn.w - learn_rate * grad_cost_w_total
                self.nn.b = self.nn.b - learn_rate * grad_cost_b_total

    def test(self, x_test, label_test, n_test_examples:int):
        '''
        Testing method.

        input format:
            x_test[number of training example] = np.array column vector
            label_test[number of training example] = 0,...,9
        '''
        error_list=[]
        n_right_predictions = 0

        for n_test in range(n_test_examples):
            # one-hot column vector enconding of label_input:
            y_label = self.one_hot_vector(label_test[n_test])

            y_prediction, prediction = self.nn.prediction(x_test[n_test])
            # calculates error of one training example
            error = np.linalg.norm(y_prediction-y_label)/2.
            if prediction == label_test[n_test]:
                n_right_predictions += 1

        accuracy = float(n_right_predictions)/n_test_examples
        return error_list, accuracy
