# Model Description

### Steps followed to build the model
##### 1. Data preparation
- 1.1 Load the data: The train.txt file is loaded from the directory and the labels and images are separted from them. The images which contains label A was in the form of A/pic_0000jpg which was also converted in the form of .jpg while loading it from the .txt file.
- 1.2 After that a panda frame was created to store the file names and the corresponding labels.
- 1.2 Labelling: The images present in the train/A, train/B, train/C, train/D was opened and the corresponding labels were added to it. Each of the image was also converted to its corresponding vectro form.
- 1.3 Normalization: I performed a grayscale normalization to reduce the effect of illumination's differences. Moreover the CNN converge faster on [0..1] data than on [0..255].
- 1.4 Label encoding: All the labels was encoded to its corresponding hot vector.
- 1.5 Split training and validation set: The training and validation data was split with test_size = 0.2.
### 3  Building the model CNN
#### Define the model

I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(4,activation="softmax")) the net outputs distribution of probability of each class.

#### Data Augmentation

As we have only small number of training examples so in order to avoid overfitting problem, we need to expand artificially our images. We can make your existing dataset even larger. The idea is to alter the training data with small transformations.

Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.

By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.


#### Setting the optimizer

Once our layers are added to the model, we need to set up a score function, a loss function and an optimisation algorithm.

We define the loss function to measure how poorly our model performs on images with known labels. It is the error rate between the oberved labels and the predicted ones. We use a specific form for categorical classifications (>2 classes) called the "categorical_crossentropy".

The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.

I choosed RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. We could also have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than RMSprop.

The metric function "accuracy" is used is to evaluate the performance our model. This metric function is similar to the loss function, except that the results from the metric evaluation are not used when training the model (only for evaluation).

#### Evaluating the model

After training the model I plotted a graph of accuracy and loss for both the validation and training set, to get more insights about how the loss function is changing over no of epocs. I train the model for 30 epochs at the model gave accuracy of 69%. The maximum accuracy was 73%.

#### Testing the model

After training the model I loaded all the image files from the folder test and converted into RGB form for prediction. The labels were predicted for the images and at last it was converted to pandas data frame for seeing the results for each of the test set images.




