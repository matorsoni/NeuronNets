from neuron_network import *
from trainer import *

from mnist import MNIST

def main():
    nn = NeuronNetwork(input_size=784, output_size=10, hidden_layers=[15])
    input = np.random.randn(784)
    dic = nn.prediction(input, print_result=True)

    # read data into variables
    # x_train[0 - 59999][0 - 783], labels_train[0 - 59999]
    mndata = MNIST('../data')
    x_train, labels_train = mndata.load_training()
    print('MNIST training data has been read')





if __name__ == "__main__":
    main()
