# Package Description

## Intro
deep_learn package contains numpy of implementation of a standard feed forward neural network. The model class `deep_learn.nn.ann` which implements feed forward neural network takes in network architecture as object field. The class method `fit` which trains the model takes train and test data and neural network hyperparameters as arguments. Here every mathematical operation is implemented using numpy to speed up computation. Other than numpy and built-in packages no other specialized libraries are used to build package.

## Folder Description
- deep_learn - contains the package which contains the class module of feed forward neural network
- analysis - contains jupyter notebooks that shows example demonstrations of deep learning tasks using the deep_learn package, it also contains the iris_classification notebook which trains neural network to classify iris data (check the notebook)
- tests - contains `test.py` which carries out unit tests of some methods of the `deep_learn` package. I put the test files outside of the package because, before building the package I planned to do some unit tests against functions of established libraries without tediously building test cases which is time consuming. For example: testing my implementation of logloss function against logloss function of sklearn. Putting the test files outside the package makes sure we dont have to install sklearn as a requirement to install this package.

## Installation Procedure
Download the package. Open the terminal, and change current directory to the directory containing the `setup.py` file. Install the package using the following command:
```sh
pip install .
```
`Note:` the jypyter notebooks and `test.py` file can run without pip installation of the package, since these files at the beginning append the path of the package to environment system path if the package is not pip installed.

## Example Usage
To create an object of the ann model write the following code:
```sh
# import the package
from deep_learn.nn import ann
# size of the layers including the input and output
layers_dims = [4,4,8,8,4,3]
# create model with given architecture
model = ann(layers_dims=layers_dims)
```

To fit the ann model write the following code:
```sh
model.fit(X_train, Y_train, X_test, Y_test, batch_size,
          learning_rate = learning_rate,
          num_iterations = num_iterations, print_cost=True, random_seed = 0)
```

**Read the jupyter notebooks in analysis folder to further understand the usage of the package**
