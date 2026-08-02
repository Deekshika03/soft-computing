# Experiment 2.1
# AIM: Build a Convolutional Neural Network (CNN) using MNIST Dataset

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

# Input image dimensions
img_rows, img_cols = 28, 28

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape images to (samples, height, width, channels)
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32")
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32")

# Normalize pixel values (0-255 → 0-1)
x_train /= 255
x_test /= 255

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Convert class labels into one-hot encoded vectors
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build CNN Model
model = Sequential()

# First Convolution Layer
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))

# Second Convolution Layer
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))

# Max Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout Layer
model.add(Dropout(0.25))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Display model architecture
model.summary()

# Compile Model
model.compile(
    loss='categorical_crossentropy',
    optimizer='Adadelta',
    metrics=['accuracy']
)

# Train Model
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

# Evaluate Model
score = model.evaluate(x_test, y_test, verbose=0)

print("\nTest Loss:", score[0])
print("Test Accuracy:", score[1])
