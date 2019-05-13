# Aerial Cactus Identification Using CNN

This project aims at building a CNN model with TensorFlow using a dataset of 17500 images of cactus to predict whether a low-resolution image contains a cactus. 

## Architecture of CNN

The first is the tensorboard graph of CNN. The second one is the flowchart of CNN using VALID as padding. This model consists of one input layer, two convolutional layers, one pooling layers, one fully connected layer accompanied by one dropout layer, and one output layer. 

![alt text] (tensorboard_graph.png)

For the The height, width and chanels of the input is 32*32*3.  The first convolutional layer has 36 filters and the second layer has 72 filters. For both convolutional layers, the kernel size is three, stride is one, padding is SAME, and Relu is used as the activation function. In the max pooling, a kernel of 2*2 and a stride of 2*2 are adopted to reduce the dimensionality of feature maps to the half. Then, the outputs from the pooling layer are reshaped back to 1-d vector for the fully connected layer. The fully connected layer has 128 units and uses Relu as the activation function. In order to decrease the overfitting, the dropout layer with the dropout rate of 0.5 is applied. The number of outputs in the output layer is two because of two classes. By using the sigmoid activation function, their probability can be obtained. Finally, the cross entropy is used to calculate the loss and the Adam optimizer is used to optimized the parameters (weights and bias) of the model. 

<img width="241" alt="Screen Shot 2019-04-07 at 9 44 55 AM" src="https://user-images.githubusercontent.com/42804316/57630424-91d33800-756b-11e9-8978-3db12e98cfc4.png">




