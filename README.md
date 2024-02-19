# Cnn

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

    Note :
        Error = is the total error from our function (Network)
        Target = is the correct label of our function
        Output = is the predicted label of our function


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
