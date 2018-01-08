# DeepDog

The purpose of DeepDog is to learn about neural networks and Tensorflow.  DeepDog uses the [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset
to train various kinds of neural networks in order to classify dog breeds based on images.

![alt text](http://vision.stanford.edu/aditya86/ImageNetDogs/images/n02106662-German_shepherd/n02106662_14247.jpg)

An image of a German Shepherd from the data set.

## Best Test Accuracies (10k iterations of mini batch gradient descent):
All loss functions are cross entropy.
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

### Convolutional Neural Networks

#### 6x6 convolution with 6 outputs, 5x5 convolution with 8 outputs, and 4x4 convolution with 12 outputs, stride of 2 instead of 2x2 max pooling, a fully connected ReLU layer with 128 weights, and no dropout
Top-1 Accuracy: 7.0%, Top-5 Accuracy: 21% (Memorized training set)

#### 6x6 convolution with 6 outputs, 5x5 convolution with 8 outputs, and 4x4 convolution with 12 outputs, stride of 2 instead of 2x2 max pooling, a fully connected ReLU layer with 128 weights, and dropout with keep probability of 0.75
Top-1 Accuracy: 8.0%, Top-5 Accuracy: 22% (Memorized training set)

#### 6x6 convolution with 6 outputs, 5x5 convolution with 8 outputs, and 4x4 convolution with 12 outputs, stride of 2 instead of 2x2 max pooling, a fully connected ReLU layer with 128 weights, and dropout with keep probability of 0.5
Top-1 Accuracy: 8.0%, Top-5 Accuracy: 25% (Memorized training set)

#### 5x5 convolution with 8 outputs, 2x2 max pooling, a fully connected ReLU layer with 32 weights, and dropout with keep probability of 0.5
Top-1 Accuracy: 8.87%, Top-5 Accuracy: 24.6% (Memorized training set)

#### Increased image size to 128x128 RGB images, all previous images were 64x64 RGB.  Convolutional net inspired by this [VGG paper](https://arxiv.org/pdf/1409.1556.pdf).  3x3 convolutional filters with 64, 128, 256, 512 weight layers.  Each weight layer is followed by a 2x2 max pooling layer.  The final 512 conv layer is followed by a ReLU fully connected layer with 1024 weights.  Dropout with keep probability 0.5.  Final output layer is softmax.   
Top-1 Accuracy: 19.65%, Top-5 Accuracy: 44.32% (Memorized training set at iteration 4600)

#### Same network layer as above network, added random mirroring (horizontal flipping from left to right) to training set mini batches
Top-1 Accuracy: 26.04%, Top-5 Accuracy: 55.00% (Memorized training set at iteration 8050)

#### Same network layer as above network, added random cropping of images in training set mini batches.  Original images were 150x150 and cropped to random 128x128 sections. Never memorized training set after 10k iterations.
Top-1 Accuracy: 30.55%, Top-5 Accuracy: 61.37%
