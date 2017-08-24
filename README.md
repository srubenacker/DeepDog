# DeepDog

The purpose of DeepDog is to learn about neural networks and Tensorflow.  DeepDog uses the [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset
to train various kinds of neural networks in order to classify dog breeds based on images.

![alt text](http://vision.stanford.edu/aditya86/ImageNetDogs/images/n02106662-German_shepherd/n02106662_14247.jpg)

An image of a German Shepherd from the data set.

## Best Test Accuracies (10k iterations of batch gradient descent):

### Simple Feed Forward Network:

#### 1 input layer, 1 softmax output layer.  No hidden layers:
Top-1 Accuracy: 5.3%, Top-5 Accuracy: 16.1%

### Feed Forward Network with Hidden Layers:

#### 1 input layer, 1 Leaky ReLU hidden layer, 1 softmax output layer:
Top-1 Accuracy: 6.7%, Top-5 Accuracy: 18.4%

#### 1 input layer, 2 Leaky ReLU hidden layer, 1 softmax output layer:
Top-1 Accuracy: 8.2%, Top-5 Accuracy: 24.1%

#### 1 input layer, 2 Leaky ReLU hidden layer with dropout, 1 softmax output layer:
Top-1 Accuracy: 9.3%, Top-5 Accuracy: 26.6%

#### 1 input layer, 3 Leaky ReLU hidden layer with dropout, 1 softmax output layer:
Top-1 Accuracy: 9.0%, Top-5 Accuracy: 27%

#### Now on to try conv nets...
