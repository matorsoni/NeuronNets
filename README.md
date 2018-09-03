# NeuronNets

Implementation of a feed forward NN for the classic MNIST classification and an implementation of a LSTM for time series analysis.
All implementations use numpy.

## LSTM
Implementation of a LSTM with peepholes connection, see the references folder.
		
The LSTM network may have many lstm blocks like the one shown in lstm/references/lstm_block_structure.png, all 
stacked one on top of each other, though this implementation only allows one block networks.
Getting the MNIST dataset: https://github.com/sorki/python-mnist