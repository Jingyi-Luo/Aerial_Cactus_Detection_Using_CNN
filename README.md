# Aerial Cactus Detection Using CNN in TensorFlow

This project aims at building a CNN model with TensorFlow using a dataset of 17500 images of cactus to predict whether a low-resolution image contains a cactus. 

[Data](https://www.kaggle.com/c/aerial-cactus-identification/data)

## Architecture of CNN

The first is the tensorboard graph of CNN. The second one is the flowchart of CNN using VALID as padding. This model consists of one input layer, two convolutional layers, one pooling layers, one fully connected layer accompanied by one dropout layer, and one output layer.

<img width="409" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57631001-a5cb6980-756c-11e9-9b58-b02f9488470f.png"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img width="241" alt="Screen Shot 2019-04-07 at 9 44 55 AM" src="https://user-images.githubusercontent.com/42804316/57630424-91d33800-756b-11e9-8978-3db12e98cfc4.png">

**Detailed dimension description**:
* Input layer: the height (H), width (W), and channels of each colorful image 32 by 32 by 3.
* The output of the first convolutional layer: the filter size (F) is 3, the striding (S) is 1, and the number of filters is 36, so the output dimension  is 30 by 30 by 36 (Equation used: (W-F)/S +1= (32-3)/1 +1 = 30).
* The output of the second convolutional layer: the input dimension is 30 by 30 by 36, the filter size (F) is 3, the striding (S) is 1, and the number of filters is 72, so the output dimension is 28 by 28 by 72 (Equation used: (W-F)/S +1= (30-3)/1 +1 = 28).
* The output of the max-pooling layer: the input dimension is 28 by 28 by 72. The filter size (F) is two and the striding (S) is two. Max-pooling doen't change the number of channels, so the output dimension is 14*14*72 (Equation used: (W-F)/S +1= (28-2)/2 +1 = 14).
* The output of the reshaped vector for fully connected layer: its dimension is 1*14112 that is obtained from the product of 14, 14 and 72 from the output of the pooling layer.
* The output of the fully connected layer: the fully connected layer has 128 units, so its output dimension is 128.
* The output layer with the sigmoid function: the predicted classes are two, so this layer’s dimension is two.

## Results

The effect of padding (VALID, SAME), optimization (Gradient Descent, and Adam optimizer), number of convolutional layers have been investigated to increase model's accuracy. Based on the optimized architecture, the training set obtained the accuray of 100% and the loss of 0.00088, and the test set obtained the accuracy of 99.7% and and the loss of 0.0096 as shown below.

Accuracy For Training and Testing Data (tensorboard graph with epochs):
<img width="560" alt="accuracy_train_valid" src="https://user-images.githubusercontent.com/42804316/57632966-b7167500-7570-11e9-9cbb-782215854ce7.png">  

Loss For Training and Testing Data (tensorboard graph with epochs):
<img width="573" alt="loss_train_valid" src="https://user-images.githubusercontent.com/42804316/57633065-e2995f80-7570-11e9-9517-3f174371f0c7.png">

----------------------------------------
The project comes from Kaggle competition. When use this model to predict a fresh new dataset from Kaggle, the accuracy reaches 99% and ranks 195 among all the competitors (April, 2019)

Snapshot from Kaggle Competition:
![195_0 99_kaggle](https://user-images.githubusercontent.com/42804316/57634270-fba31000-7572-11e9-9634-5491a9fe7780.png)









