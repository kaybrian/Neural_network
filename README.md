# Simple Neural Network Implementation

This repository contains a simple implementation of a neural network in Python from scratch. The neural network is designed to perform binary classification based on a small training dataset.

## Overview

The neural network implemented here is a basic feedforward neural network with one hidden layer. It is trained using a backpropagation algorithm to minimize the error between predicted and target outputs. The network uses a sigmoid activation function in the hidden layer for non-linearity.


## Usage

To use the neural network, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have Python installed.
3. Open a terminal and navigate to the repository directory.
4. Run the `neural_network.py` file.

## Files

- `neural_network.py`: Contains the implementation of the neural network.
- `README.md`: This file, providing an overview of the repository and instructions for usage.


## Training Data

The training data used by the neural network is pre-coded within the script. It consists of four samples, each with three input features and a corresponding binary output.

## Neural Network Class

The `NeuralNetwork` class is the core of this implementation. It includes methods for initialization, forward propagation (`think`), training, and utility functions for the sigmoid activation and its gradient.


## Example

```python
neural_network = NeuralNetwork()
print("Random Starting weights: ", neural_network.weights)

# Train the neural network
neural_network.train(training_data, number_of_iterations=10000)

# Print the new training weights
print("Updated Weights: ", neural_network.weights)

# Make predictions with new data
new_data = [0, 1, 0]
prediction = neural_network.think(new_data)

# Print the prediction
print("New Data Prediction: ", prediction)
```



# Background Information here (Theory for the code implementation)

# Activation Function

For a neuron Input X, Weight W and Bias B, the output is:
```python3
    for input in inputs:
        return Sum(Input * Weight) + Bias
```


# Activation Function
```python3
    Output = (1 / (1 + exp(-Input)))
```


# Training process
First we assign random numbers to our weights


# Error total error in the network
```python3
    Error = 1/2 * (Target - Output)^2
```

```python3

Note :
    Error = is the total error from our function (Network)
    Target = is the correct label of our function
    Output = is the predicted label of our function
```


# Adjusting the weight
LearningRate = also known as the Gradient Descent finds the minimum by taking steps proportional to the negative of the gradient.

Input = Input
Target = Target
Output = Output

## getting the Error cost
```python3
    Error cost function Gradient = - Input * ErrorInOutput * SigmoidCurveGradient
```

## Adjusting the weight
```python3
    Weight adjust = Input * ErrorInOutput * SigmoidCurveGradient
```

## SigmoidCurveGradient
```python3
sigmoidGradient = neuronOutput * (1 - neuronOutput)
```

## FInal OutPut Formulas are
```python3
    Weight adjust = Input * ErrorInOutput * SigmoidCurveGradient

    ## where
    sigmoidGradient = neuronOutput * (1 - neuronOutput)

```
