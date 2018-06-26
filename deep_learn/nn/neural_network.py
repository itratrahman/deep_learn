import numpy as np
from ..utils.activation import sigmoid, relu
from ..utils.cost import logistic_cost, logloss
from ..utils.back_prop import sigmoid_backward, relu_backward


class ann(object):


    """a class module which implements feed forward neural network"""
    def __init__(self, layers_dims):

        self.layers_dims = layers_dims
        self.parameters = None
        self.costs = None


    @staticmethod
    def initialize_parameters_deep(layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
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
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache


    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
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
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = ann.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "sigmoid")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = ann.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)

        # assert(AL.shape == (1,X.shape[1]))

        return AL, caches


    @staticmethod
    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
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

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
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

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = ann.linear_activation_backward(dAL, current_cache, activation = "sigmoid")

        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
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

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters

    @staticmethod
    def predict_binary(probas):

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

        p = np.argmax(probas, axis=0)

        p = p.reshape(-1,1).T

        return p


    @staticmethod
    def predict(X, parameters):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
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

        m = X.shape[1]

        y_ = ann.predict(X, parameters)

        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        binary = y.shape[0] == 1
        if binary:
            print("Accuracy: "  + str(np.sum((y_ == y)/m)))
        else:
            y = np.argmax(y, axis=0)
            y = y.reshape(-1,1).T
            print("Accuracy: "  + str(np.sum((y_ == y)/m)))



    def fit(self, X_train, Y_train, X_test, Y_test, batch_size, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, random_seed = 1):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(random_seed)
        costs = []                         # keep track of cost

        n_classes = Y_train.shape[1]

        # Parameters initialization. (≈ 1 line of code)
        ### START CODE HERE ###
        parameters = ann.initialize_parameters_deep(self.layers_dims)
        ### END CODE HERE ###

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            rand_index = np.random.choice(X_train.shape[0], size=batch_size)
            X_rand = X_train[rand_index].T
            Y_rand = Y_train[rand_index].T

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = ann.L_model_forward(X_rand, parameters)
            ### END CODE HERE ###

            # Compute cost.
            ### START CODE HERE ### (≈ 1 line of code)
            if n_classes == 1:
                cost = logistic_cost(AL, Y_rand)
            else:
                cost = logloss(AL, Y_rand)
            ### END CODE HERE ###

            # Backward propagation.
            ### START CODE HERE ### (≈ 1 line of code)
            grads = ann.L_model_backward(AL, Y_rand, caches)
            ### END CODE HERE ###

            # Update parameters.
            ### START CODE HERE ### (≈ 1 line of code)
            parameters = ann.update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        self.parameters = parameters
        self.costs = costs

        accuracy = ann.accuracy(X_test.T, Y_test.T, parameters)
