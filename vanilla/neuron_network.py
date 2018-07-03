import numpy as np

class functions:

    def sigmoid(z):
        return 1./(1.+np.exp(-z))

    def d_sigmoid(z):
        return functions.sigmoid(z)*(1.-functions.sigmoid(z))

    def relu(z):
        return 0. if z<=0. else z

    def d_relu(z):
        return 0. if z<=0. else 1.

    def choose(func_name = 'sigmoid'):
        if func_name == 'sigmoid':
            return functions.sigmoid, functions.d_sigmoid
        if func_name == 'relu':
            return functions.relu, functions.d_relu

class NeuronNetwork:

    def __init__(self, input_size=784, output_size=10, hidden_layers=[], activation='sigmoid'):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.act, self.d_act = functions.choose(activation)
        self.n_layers = len(self.layers)

        # Random initialization of weights and biases
        # Format: w[l][j][k] = weight of k-th neuron in layer l to j-th neuron in layer l+1
        # b[l][j] = bias of j-th neuron in layer l+1
        # L=len(self.layers), therefore l = 0,...,L-2 = range(L-1)
        w_list = []
        b_list = []
        L=len(self.layers)
        for l in range(L-1):
            # assings float64 type to this random initialization
            w = np.zeros((self.layers[l+1], self.layers[l]), dtype=np.float64)
            w[:] = np.random.randn(*w.shape)
            w_list.append(w)

            b = np.zeros((self.layers[l+1],1), dtype=np.float64)#.reshape((self.layers[l+1],1))
            b[:] = np.random.randn(*b.shape)
            b_list.append(b)

        self.w = np.array(w_list)
        self.b = np.array(b_list)

    def prediction(self, x_input, return_type='array',print_result = False):
        # x_input expects a row np.array
        assert x_input.shape==(self.layers[0],), "Incompatible input format"
        L = len(self.layers)

        #calculates the output vector
        y_prediction = x_input.reshape(self.layers[0],1) # turns input into column vector
        for l in range(L-1):
            y_prediction = self.act( np.dot( self.w[l], y_prediction ) + self.b[l] )
        # prediction:
        prediction = y_prediction.max()

        if print_result:
            for i in range(self.layers[-1]):
                print(
                    "{}: {}".format(i, round(y_prediction[i][0]*100,2) )
                )

        if return_type=='array':
            return y_prediction, prediction

        if return_type=='dictionary':
            result_dict = {} #dictionary with 0,...,9 as keys and y_prediction[i] as values
            for i in range(self.layers[-1]):
                result_dict[i] = y_prediction[i][0]
            return result_dict, prediction
