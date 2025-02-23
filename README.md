# Deeplearning_projects
Decoding handwritten digits using AI
# Neural Network for MNIST Classification
## Overview
This project implements a simple neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained and evaluated to achieve high accuracy in recognizing digits from 0 to 9.

## Installation
Ensure you have Python installed. Install the required dependencies using:
```sh
pip install tensorflow numpy matplotlib
```

## Dataset
The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Model Architecture
The model consists of the following layers:
- **Flatten Layer**: Converts the 28x28 image into a 1D array of 784 elements.
- **Dense Layer** (128 neurons, ReLU activation): Fully connected layer.
- **Output Layer** (10 neurons, Softmax activation): Outputs probabilities for each digit (0-9).

## Code Structure

### Load the Dataset
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

### Preprocess the Data
```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

### Define the Model
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

### Compile the Model
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Train the Model
```python
history = model.fit(x_train, y_train, epochs=10)
```

### Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
```

## Results
The model achieves approximately **97-99% accuracy** on the test dataset.

## Visualization
### Example training image
```python
plt.matshow(x_train[8], cmap='gray')
plt.show()
```

### Training loss over epochs
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Vs Epochs')
plt.show()
```

## Predictions
To make predictions on test data:
```python
predictions = model.predict(x_test)
print(np.argmax(predictions[0]))  # Predicted class
print(y_test[0])  # Actual class
```

## Conclusion
This project successfully implements a neural network for handwritten digit recognition using TensorFlow and Keras. The model demonstrates high accuracy and effective classification performance on the MNIST dataset.

