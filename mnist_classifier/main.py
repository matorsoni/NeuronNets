from neuron_network import *
from trainer import *

#from mnist import MNIST
## little hack to make the import work: copy loader.py into this folder and
## import MNIST class from there
from loader import MNIST

def normalize_data(train_list_in, test_list_in):
    '''
    Input:
        train_list_in[i] = list of pixel values
        test_list_in[i] = list of pixel values

    Output:
        train_list_out[i] = np.array column vector
        test_list_out[i] = np.array column vector
    '''
    train_list_out = [np.array(values).reshape(len(values),1)/255. for values in train_list_in]
    test_list_out = [np.array(values).reshape(len(values),1)/255. for values in test_list_in]
    return train_list_out, test_list_out


def main():
    nn = NeuronNetwork(input_size=784, output_size=10, hidden_layers=[15])
    #input = np.random.randn(784).reshape(784,1)
    #dic = nn.prediction(input, print_result=True)

    # read data into variables
    # x_train[0 - 59999][0 - 783], labels_train[0 - 59999]
    mndata = MNIST('../data')
    x_train_in, labels_train = mndata.load_training()
    print('MNIST training data has been read')
    x_test_in, labels_test = mndata.load_testing()
    print('MNIST test data has been read')
    x_train, x_test = normalize_data(x_train_in, x_test_in)
    print('MNIST data has been normalized')

    trainer = Trainer(nn)
    # train(n_training_examples=60000, batch_size=200, n_epochs=20, learn_rate=1.5) = 0.872 accuracy
    # train(n_training_examples=60000, batch_size=200, n_epochs=40, learn_rate=1.5) = 0.906 accuracy
    trainer.train(x_train, labels_train, n_training_examples=60000, batch_size=200, n_epochs=50, learn_rate=1.5)
    error_list, acc = trainer.test(x_test, labels_test, n_test_examples=1000)

    #print ('error: {} ----> {}'.format(error_list[0], error_list[-1]))
    print ('accuracy = {}'.format(acc))

    #testing with examples

    for i in range(10):
        vec, pred = nn.prediction(x_test[i])
        print( 'Image: {} ====> Prediction: {}'.format(labels_test[i], pred))







if __name__ == "__main__":
    main()
