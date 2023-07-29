# First we import some libraries that will help us later:

# tensorflow and keras - these let us build and train neural networks
# numpy - helps with numerical and math operations
# matplotlib - helps display images and graphs
# Next we load a dataset called Fashion MNIST that has 70,000 images of clothes in 10 categories like shirts, dresses, shoes etc.

# We split the data into training images and testing images.

# The training images will be used to train the neural network model. The testing images will be used after training to evaluate how good the model is.

# We scale all the image pixels to be between 0 and 1 to make the data easier to work with.

# Then we define our neural network model to have:

# A flatten layer that turns each 2D 28x28 image into a 1D list of numbers
# A dense layer with 128 nodes using the 'relu' activation. This finds patterns in the input data.
# A final output layer with 10 nodes to match the 10 clothing categories, using 'softmax' activation to predict which category has the highest probability for each image.
# We compile the model to configure the training process. We use the 'adam' optimizer and 'categorical_crossentropy' loss function which are common choices.

# Next we train the model by showing it the training images and labels for 5 epochs. An epoch is one pass through all the training data.

# We evaluate the trained model on the test images and print out its accuracy.

# Finally we use the model to make predictions on a few test images and display the predictions next to the actual label. This helps us visualize how well the model is classifying the clothes.

# So in summary, we loaded image data, defined a neural network architecture, trained it on the data, and evaluated its accuracy - all using just a few lines of code!

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()