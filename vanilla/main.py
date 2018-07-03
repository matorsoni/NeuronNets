from neuron_network import *
from trainer import *

#from mnist import MNIST
## little hack to make the import work: copy loader.py into this folder and
## import MNIST class from there
from loader import MNIST

def normalize_data():
    pass


def main():
    nn = NeuronNetwork(input_size=784, output_size=10, hidden_layers=[15])
    input = np.random.randn(784)
    dic = nn.prediction(input, print_result=True)
    print(nn.n_layers)

    # read data into variables
    # x_train[0 - 59999][0 - 783], labels_train[0 - 59999]
    mndata = MNIST('../data')
    x_train, labels_train = mndata.load_training()
    print('MNIST training data has been read')
    x_test, labels_test = mndata.load_testing()
    print('MNIST test data has been read')

    #min max scaling
    print(np.array(x_train[0])/255.)

    trainer = Trainer(nn)
    trainer.train(x_train, labels_train, n_training_examples=5000, batch_size=100, n_epochs=1, learn_rate=1.)
    error_list, acc = trainer.test(x_test, labels_test, n_test_examples=1000)

    print ('error: {} ----> {}'.format(error[0], error[-1]))
    print ('accuracy = {}'.format(acc))







if __name__ == "__main__":
    main()
