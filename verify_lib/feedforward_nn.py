import numpy as np

# Input data
X = np.array([
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1]
])

# Actual output
y = np.array([
    [1],
    [1],
    [0]
])

print("Input:")
print(X)

print("\nActual Output:")
print(y)


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def derivative_sigmoid(x):
    return x * (1 - x)


# Parameters
epoch = 5000
learning_rate = 0.1

input_neurons = X.shape[1]
hidden_neurons = 3
output_neurons = 1


# Initialize weights and biases
np.random.seed(42)

wh = np.random.uniform(
    size=(input_neurons, hidden_neurons)
)

bh = np.random.uniform(
    size=(1, hidden_neurons)
)

wout = np.random.uniform(
    size=(hidden_neurons, output_neurons)
)

bout = np.random.uniform(
    size=(1, output_neurons)
)


# Training
for i in range(epoch):

    # ---------------- FORWARD PROPAGATION ----------------

    # Input to hidden layer
    hidden_input = np.dot(X, wh) + bh

    # Activation of hidden layer
    hidden_output = sigmoid(hidden_input)

    # Hidden layer to output layer
    output_input = np.dot(hidden_output, wout) + bout

    # Activation of output layer
    output = sigmoid(output_input)


    # ---------------- BACKPROPAGATION ----------------

    # Calculate error
    error = y - output

    # Derivative of output
    slope_output = derivative_sigmoid(output)

    # Derivative of hidden layer
    slope_hidden = derivative_sigmoid(hidden_output)

    # Output layer gradient
    d_output = error * slope_output

    # Error at hidden layer
    error_hidden = d_output.dot(wout.T)

    # Hidden layer gradient
    d_hidden = error_hidden * slope_hidden


    # Update output weights and bias
    wout += hidden_output.T.dot(d_output) * learning_rate
    bout += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    # Update hidden weights and bias
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate


# Final output
print("\nOutput from the model:")
print(output)

print("\nPredicted Output:")
print((output > 0.5).astype(int))
