
import pandas as pd
import numpy as np

# Sigmoid activation function
def logistic(x):
    return 1.0 / (1 + np.exp(-x))

# Derivative of sigmoid
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

# Parameters
LR = 0.1
I_dim = 3
H_dim = 4
epoch_count = 100

# Initialize weights
np.random.seed(1)
weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

# Training dataset
training_data = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

target_output = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Training using forward and backward propagation
for epoch in range(epoch_count):

    for sample in range(len(training_data)):

        # Forward propagation
        preActivation_H = np.dot(training_data[sample], weights_ItoH)
        postActivation_H = logistic(preActivation_H)

        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)

        # Calculate error
        FE = postActivation_O - target_output[sample]

        # Backpropagation
        S_error = FE * logistic_deriv(preActivation_O)

        for H_node in range(H_dim):

            gradient_HtoO = S_error * postActivation_H[H_node]

            for I_node in range(I_dim):

                gradient_ItoH = (
                    S_error * weights_HtoO[H_node]
                    * logistic_deriv(preActivation_H[H_node])
                    * training_data[sample, I_node]
                )

                weights_ItoH[I_node, H_node] -= LR * gradient_ItoH

            weights_HtoO[H_node] -= LR * gradient_HtoO

# Testing
correct = 0

for sample in range(len(training_data)):

    hidden = logistic(np.dot(training_data[sample], weights_ItoH))
    output = logistic(np.dot(hidden, weights_HtoO))

    predicted = 1 if output > 0.5 else 0

    print("Actual:", target_output[sample],
          "Predicted:", predicted)

    if predicted == target_output[sample]:
        correct += 1

# Accuracy
accuracy = (correct / len(training_data)) * 100

print("\nPercentage of correct classifications:")
print(accuracy, "%")
