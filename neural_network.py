import math
import random


# Training Dataset, Pre-coded in the code

training_data = [
    {"inputs": [0, 0, 1], "Outputs": 0},
    {"inputs": [1, 1, 1], "Outputs": 1},
    {"inputs": [1, 0, 1], "Outputs": 1},
    {"inputs": [0, 1, 1], "Outputs": 0},
]

class NeuralNetwork:
    '''
    A description of the class, its attributes, and its methods.
    '''
    def __init__(self):
        '''
            Constructor of the class Neural
        '''
        # get same random numbers
        random.seed(1)

        # get 3 random values
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
    def think(self, neuron_inputs):
        """
         A description of the entire function, its parameters, and its return types.
        """
        sum_of_weights = self.__sum_of_weighted_inputs(neuron_inputs)
        neuron_output = self.__sigmoid(sum_of_weights)
        return neuron_output

    def train(self, training_data_set, number_of_iterations):
        """
            Function to train the neural network
        """
        for iteration in range(number_of_iterations):
            for train_data in training_data_set:
                #  predict the output based on the training data inputs
                predicted_output = self.think(train_data["inputs"])

                # calculate the error btn the target and the predicated output
                error_in_output = train_data["Outputs"] - predicted_output

                # update the weights
                for index in range(len(self.weights)):
                    # access the Neuron inputs
                    neuron_input = train_data["inputs"][index]

                    # calculate the how much we need to update the weight
                    updated_weight = neuron_input * error_in_output * self.__sigmod_gradient(predicted_output)

                    # update the weight
                    self.weights[index] += updated_weight


    def __sigmoid(self, sum_of_weights):
        '''This is is the sigmoid function'''
        return 1 / (1 + math.exp(-sum_of_weights))

    def __sigmod_gradient(self, neuron_output):
        '''This is the gradient of the sigmoid function'''
        return neuron_output * (1 - neuron_output)

    def __sum_of_weighted_inputs(self, neuron_inputs):
        '''This is the sum of the weighted inputs'''
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input

        return sum_of_weighted_inputs

neural_network = NeuralNetwork()
print("Random Starting weights: ", neural_network.weights)

# train the neural network now
neural_network.train(training_data, number_of_iterations= 10000)

# print the new training weights
print("Random Starting weights: ", neural_network.weights)


# make the predictions now with the new data
new_data = [0,1,0]

predication = neural_network.think(new_data)

# print the new predictions
print("New data prediction: ", predication)
