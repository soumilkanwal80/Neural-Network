
# 3 Layer Feed Forward Neural Network with BackPropagation, Gradient Descent, Dropout

import numpy as np 
np.set_printoptions(suppress=True)

#sigmoid
def sigmoid(x):
	return 1/(1+np.exp(-x))

#derivative
def derivative(x):
	return x*(1-x)

#Dropout
dropout = 0.2
do_dropout = True

#Learning Rate
alpha = 5

#Dimension of Hidden Layer
hidden_layer = 32

#input
X=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

#output
y=np.array([[0, 0, 1, 1]]).T

#seed random rumbers to make calculations
np.random.seed(1)

#Initialize Weights
syn0 = 2*np.random.random((3, hidden_layer)) - 1 
syn1 = 2*np.random.random((hidden_layer, 1)) - 1

for iter in range(10000):
	#Forward Propagation
	l0=X;
	#Activating First Layer
	l1=sigmoid(np.dot(l0, syn0))
	#Dropout
	if(do_dropout):
		l1 *= np.random.binomial([np.ones((len(X), hidden_layer))], 1-dropout)[0]*(1.0/(1-dropout))

	#Activating Second Layer	
	l2=sigmoid(np.dot(l1, syn1))


	#Backpropagation
	#Calculate Bias
	l2_error = y-l2

	#Multiply error by derivative of sigmoid
	l2_delta = l2_error * derivative(l2)

	#Error in layer 1 due to layer 2
	l1_error = np.dot(l2_delta, syn1.T)

	#Multiply error by derivative of sigmoid
	l1_delta = l1_error * derivative(l1)

	#update weights
	syn1 += alpha * (np.dot(l1.T, l2_delta))
	syn0 += alpha * (np.dot(l0.T, l1_delta))



print("Prediction")
print(l2)
	