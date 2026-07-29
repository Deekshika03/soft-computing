import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Step 1: Load the MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Training Images Shape:", train_images.shape)
print("Training Labels Shape:", train_labels.shape)

# Step 2: Preprocess the Data
train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Step 3: Build the ANN Model
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print("\nModel Summary:")
model.summary()

# Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train the Model
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# Step 6: Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\n==============================")
print("Test Accuracy:", round(test_acc * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))
print("==============================")

# Step 7: Predict One Test Image
(_, _), (test_images_original, test_labels_original) = mnist.load_data()

index = 10
image = test_images_original[index]

plt.imshow(image, cmap='gray')
plt.title(f"Actual Label: {test_labels_original[index]}")
plt.axis('off')
plt.show()

image = image.reshape(1, 784).astype("float32") / 255
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Prediction Probabilities:\n")
print(prediction)
