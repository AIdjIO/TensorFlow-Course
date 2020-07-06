#useful link to find various dataset for different ML projects.
https://blog.cambridgespark.com/50-free-machine-learning-datasets-image-datasets-241852b03b49

import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#additionally you may have to install CUDA and CUdnn (versions to download depend on tensorflow version)


#Load the dataset
fashion_mnist = keras.datasets.fashion_mnist

#Define training and testing data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Plot the first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Scale grayscale images from 0..255 to 0.0..1.0
train_images=train_images/255.0
test_images=test_images/255.0

#Define the model
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128,activation=tf.nn.relu),keras.layers.Dense(10,activation=tf.nn.softmax)])

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(train_images,train_labels,epochs=5)

#Evaluate the model
test_loss, test_acc= model.evaluate(test_images, test_labels)
print(test_acc)

#predictions of test images
predictions = model.predict(test_images)
print(predictions[0])

print(numpy.argmax(predictions[0])) #show which category was predicted by looking for the category with highest probability

#check what that test image label actually is to verify if prediction is correct
print(test_labels[0])



