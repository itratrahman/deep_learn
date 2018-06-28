# Package Description

## Intro
deep_learn package contains numpy of implementation of a standard feed forward neural network. The model class `deep_learn.nn.ann` which implements feed forward neural network takes in network architecture as object field. The class method `fit` which trains the model takes neural network hyperparameters as arguments. Here every mathematical operation is implemented using numpy to speed up computation. Other than numpy and built-in packages no other specialized libraries are used to build package.

## Folder Description
- deep_learn - contains the package which in turn contains the class module of feed forward neural network
- analysis - contains jupyter notebooks that carry out deep learning tasks using the deep_learn package, it also contains the iris_classification notebook which trains neural network to classify iris data (check the notebook)
- tests - contains test.py which carries out unit tests of some methods of the `deep_learn.nn.ann` class. I put the test file outside of the package because, before building the package I planned to do some unit tests against functions of established libraries without tediously building test cases which is time consuming. For example: testing the my implementation of logloss function with logloss function of sklearn. Putting the test filed outside the package makes sure we dont have to install sklearn as a requirement to install this package.

## Installation Procedure
Download the package. Open the terminal, and change current directory to the directory containing the **deep_learn** package. Install the package using the following command.
```sh
$ pip install .
```
The jypyter notebooks and test.py file can run without pip installing since these files at the beginning append the path of the package to python system path if the package is not installed.
