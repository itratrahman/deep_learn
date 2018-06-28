
# import modules from deep_learn package
try:
    from deep_learn.nn import ann
    from deep_learn.utils.cost import logloss
except:
    from config import *
    append_path('../')
    from deep_learn.nn import ann
    from deep_learn.utils.cost import logloss

# import statements
from test_cases import *
import numpy as np
import math
import time

class test(object):

    """Test"""

    @staticmethod
    def linear_forward_test():

        A, W, b = linear_forward_test_case()
        Z, linear_cache = ann.linear_forward(A, W, b)
        assert(np.allclose(Z, np.array([[ 3.26295337, -1.23429987]])))

        print("linear_forward_test passed", "\n")

    @staticmethod
    def linear_activation_forward_test():

        A_prev, W, b = linear_activation_forward_test_case()
        A, linear_activation_cache = ann.linear_activation_forward(A_prev, W, b, activation = "sigmoid")
        assert(np.allclose(A, np.array([[ 0.96890023, 0.11013289]])))
        A, linear_activation_cache = ann.linear_activation_forward(A_prev, W, b, activation = "relu")
        assert(np.allclose(A, np.array([[ 3.43896131 , 0.]])))

        print("linear_activation_forward_test passed", "\n")

    @staticmethod
    def L_model_forward_test():

        X, parameters = L_model_forward_test_case_2hidden()
        AL, caches = ann.L_model_forward(X, parameters)
        assert(np.allclose(AL, np.array([[ 0.55865298,  0.52006807,  0.51071853,  0.53912084]])))
        assert(len(caches)==3)

        print("L_model_forward_test passed", "\n")


    @staticmethod
    def linear_backward_test():

        # Set up some test inputs
        dZ, linear_cache = linear_backward_test_case()
        dA_prev, dW, db = ann.linear_backward(dZ, linear_cache)
        assert(np.allclose(dA_prev, np.array([[ 0.51822968,-0.19517421],[-0.40506361,0.15255393], [ 2.37496825,-0.89445391]])))
        assert(np.allclose(dW, np.array([[-0.10076895,  1.40685096,  1.64992505]])))
        assert(np.allclose(db, np.array([[ 0.50629448]])))
        print("linear_backward_test passed", "\n")


    @staticmethod
    def linear_activation_backward_test():

        dAL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = ann.linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
        assert(np.allclose(dA_prev, np.array([[ 0.11017994,  0.01105339],[ 0.09466817,  0.00949723],[-0.05743092, -0.00576154]])))
        assert(np.allclose(dW, np.array([[ 0.10266786,  0.09778551, -0.01968084]])))
        assert(np.allclose(db, np.array([[-0.05729622]])))

        dA_prev, dW, db = ann.linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
        assert(np.allclose(dA_prev, np.array([[ 0.44090989, 0.], [ 0.37883606, 0.],[-0.2298228,0.]])))
        assert(np.allclose(dW, np.array([[ 0.44513824, 0.37371418, -0.10478989]])))
        assert(np.allclose(db, np.array([[-0.20837892]])))

        print("linear_activation_backward_test passed", "\n")


    @staticmethod
    def L_model_backward_test():

        AL, Y_assess, caches = L_model_backward_test_case()
        grads = ann.L_model_backward(AL, Y_assess, caches)
        assert(np.allclose(grads['dW1'], np.array([[ 0.0944987, 0.01377483, 0.03015356, 0.02322328],
        [-0.09912664, -0.01366395, -0.03132431, -0.02417861],
        [ 0.01173121, 0.0016263, 0.00371069, 0.00286357]])))
        assert(np.allclose(grads['db1'], np.array([[-0.0357204 ],[ 0.03467639],[-0.00413664]])))
        assert(np.allclose(grads['dA1'], np.array([[0.12913162, -0.44014127],[-0.14175655, 0.48317296],[0.01663708, -0.05670698]])))

        print("L_model_backward_test passed", "\n")


    @staticmethod
    def update_parameters_test():

        parameters, grads = update_parameters_test_case()
        parameters = ann.update_parameters(parameters, grads, 0.1)
        assert(np.allclose(parameters['W1'], np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
        [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
        [-1.0535704,  -0.86128581,  0.68284052,  2.20374577]])))
        assert(np.allclose(parameters['b1'], np.array([[-0.04659241],[-1.28888275],[ 0.53405496]])))
        assert(np.allclose(parameters['W2'], np.array([[-0.55569196,  0.0354055,  1.32964895]])))
        assert(np.allclose(parameters['b2'], np.array([[-0.84610769]])))

        print("update_parameters_test passed", "\n")


    @staticmethod
    def predict_multiclass_test():

        y_ = np.array([[.01, .9, .35],[.8, .3, .1],[.1,.25,.75],[0.4, .2,.95],[.65, .001,0.06]])
        y_ = y_.T
        assert(np.allclose(ann.predict_multiclass(y_), np.array([[1, 0, 2, 2,0]])))

        print("predict_multiclass_test passed", "\n")

    @staticmethod
    def logloss_test():

        from sklearn.metrics import log_loss

        y_ = np.array([[.01, .9, .35],[.8, .3, .1],[.1,.25,.75],[0.4, .2,.95],[.65, .001,0.06]])
        y = np.array([[0, 1, 0],[1, 0, 0],[0,0,1],[0, 0,1],[1, 0,0]])
        assert(log_loss(y,y_) == logloss(y.T,y_.T))

        print("logloss_test passed", "\n")


##The main script
if __name__ == "__main__":

    print("#####################Test Results####################", "\n")

    now = time.time()

    test.linear_forward_test()

    test.linear_activation_forward_test()

    test.L_model_forward_test()

    test.linear_backward_test()

    test.linear_activation_backward_test()

    test.L_model_backward_test()

    test.update_parameters_test()

    test.predict_multiclass_test()

    test.logloss_test()

    done = time.time()

    t = done - now
    minutes = t//60
    seconds = round(t%60, 2)

    print("Time taken to complete the test: ", minutes,\
          " mins ", seconds, "secs", "\n")

    print("######################################################")
