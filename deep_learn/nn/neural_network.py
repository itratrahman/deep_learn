
# import statements
import numpy as np
from ..utils.activation import sigmoid, relu
from ..utils.cost import logistic_cost, logloss
from ..utils.back_prop import sigmoid_backward, relu_backward


class ann(object):


    """a class module which implements feed forward neural network"""

    def __init__(self, layers_dims):

        '''constructor which initializes the neural network architecture
        and contains model fields which are initialized after training'''

        self.layers_dims = layers_dims
        # contains neural network parameters
        self.parameters = None
        # list of costs during training stored at every 100th iteration
        self.costs = None
        # store the accuracy of the model
        self.accuracy = None


    @staticmethod
    def initialize_parameters_deep(layer_dims):
        """
        Arguments:
        """

        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


        return parameters


    @staticmethod
    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation
        """

        Z = W.dot(A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache


    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = ann.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = ann.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache


    @staticmethod
    def L_model_forward(X, parameters):
        """
        Implement forward propagation for L layer model
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> SIGMOID]*L. Add "cache" to the "caches" list.
        for l in range(1, L+1):
            A_prev = A
            A, cache = ann.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "sigmoid")
            caches.append(cache)

        AL = A

        return AL, caches


    @staticmethod
    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = ann.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = ann.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db


    @staticmethod
    def L_model_backward(AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initialize the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = ann.linear_activation_backward(dAL, current_cache, activation = "sigmoid")

        for l in reversed(range(L-1)):
            # lth layer: (SIGMOID -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = ann.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "sigmoid")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    @staticmethod
    def update_parameters(parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        """

        L = len(parameters) // 2 # number of layers in the neural network

        # update wieght and bias parameters
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters

    @staticmethod
    def predict_binary(probas):

        '''computes prediction for binary classification'''

        m = probas.shape[1]
        p = np.zeros((1,m))

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0

        return p

    @staticmethod
    def predict_multiclass(probas):

        '''computes prediction for multiclass classification'''

        p = np.argmax(probas, axis=0)

        p = p.reshape(-1,1).T

        return p


    @staticmethod
    def predict(X, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.
        """

        # Forward propagation
        probas, caches = ann.L_model_forward(X, parameters)

        binary = probas.shape[0] == 1
        if binary:
            p = ann.predict_binary(probas)
        else:
            p = ann.predict_multiclass(probas)

        return p


    @staticmethod
    def accuracy(X, y, parameters):

        '''computes accuracy of the model given X y data'''

        m = X.shape[1]

        y_ = ann.predict(X, parameters)

        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        binary = y.shape[0] == 1
        if binary:
            accuracy = np.sum((y_ == y)/m)
        else:
            y = np.argmax(y, axis=0)
            y = y.reshape(-1,1).T
            accuracy = np.sum((y_ == y)/m)

        return accuracy


    def fit(self, X_train, Y_train, X_test, Y_test, batch_size, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, random_seed = 0):
        """
        Implements a L-layer neural network: [LINEAR->SIGMOID]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X_train -- X data of train set, numpy array of shape (number of examples, number of featues)
        Y_train -- y data of train set, numpy array of shape (number of examples, number of classes)
        X_test -- X data of test set, numpy array of shape (number of examples, number of featues)
        Y_test -- y data of test set, numpy array of shape (number of examples, number of classes)
        batch_size -- batch size of batch gradient descent
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        random_seed -- random seed of numpy random class

        """

        # set the random seed of numpy
        np.random.seed(random_seed)
        # list to store the cost at every 100th iteration
        costs = []
        # number of classes
        n_classes = Y_train.shape[1]

        # Parameters initialization
        parameters = ann.initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, num_iterations+1):

            # randomly choose a batch
            rand_index = np.random.choice(X_train.shape[0], size=batch_size)
            X_rand = X_train[rand_index].T
            Y_rand = Y_train[rand_index].T

            # Forward propagation: [LINEAR -> SIGMOID]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = ann.L_model_forward(X_rand, parameters)

            # Compute logistic cost for binary classification
            # Compute logloss for multiclass classification
            if n_classes == 1:
                cost = logistic_cost(AL, Y_rand)
            else:
                cost = logloss(Y_rand, AL)

            # Backward propagation.
            grads = ann.L_model_backward(AL, Y_rand, caches)

            # Update parameters
            parameters = ann.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                if n_classes == 1:
                    print("Logistic cost after iteration %i: %f" %(i, cost))
                else:
                    print("Log loss after iteration %i: %f" %(i, cost))

            # Save the cost every 100 training example
            if i % 100 == 0:
                costs.append(cost)

        # compute accuracy using test set
        accuracy = ann.accuracy(X_test.T, Y_test.T, parameters)

        # print accuracy
        if print_cost:
            print("Accuracy: "  + str(accuracy))

        # set the object fields
        self.parameters = parameters
        self.costs = costs
        self.accuracy = accuracy
