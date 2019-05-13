# Aerial Cactus Identification Using CNN

This project aims at building a CNN model with TensorFlow using a dataset of 17500 images of cactus to predict whether a low-resolution image contains a cactus. 

## Architecture of CNN

The first is the tensorboard graph of CNN. This model consists of one input layer, two convolutional layers, one pooling layers, one fully connected layer accompanied by one dropout layer, and one output layer. 

The input X has a height of 32, a width of 32 and channels of 3, which is further fed into the convolutional layer. In the first convolutional layer, 36 filters are utilized, the kernel size of three is chosen, stride is one, padding is SAME, Relu is used as the activation function. In the second convolutional layer, the number of filters is 72, and other arguments are the same as the first convolutional layer. After the convolution layers, the max pooling is adopted to reduce the dimensionality of the feature maps to the half both for height and width by using a kernel of two by two and a stride of two by two. Before going to the fully connected layer, the outputs from the pooling layer are reshaped back to 1-d vector. In the fully connected layer, the number of units is 128, and the activation function is Relu. In order to decrease the overfitting, the dropout layer with the dropout rate of 0.5 is applied. The number of outputs in the output layer is two because of two classes. By using the sigmoid activation function, the probability can be obtained. Finally, the cross entropy is used to calculate the loss and the Adam optimizer is used to optimized the parameters (weights and bias) of the model. 





