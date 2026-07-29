import numpy as np

class NeuralNetwork:

    def __init__(self):
        # Initialize random weights
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network
    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(
                training_inputs.T,
                error * self.sigmoid_derivative(output)
            )
            self.synaptic_weights += adjustments

    # Predict output
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights:")
    print(neural_network.synaptic_weights)

    # Training data
    training_inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the model
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training:")
    print(neural_network.synaptic_weights)

    # User input
    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering New Situation:",
          user_input_one,
          user_input_two,
          user_input_three)

    print("New Output Data:")
    print(neural_network.think(
        np.array([user_input_one,
                  user_input_two,
                  user_input_three])
    ))

    print("Done")