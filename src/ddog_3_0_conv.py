import tensorflow as tf
import time
import ddog
import matplotlib.pyplot as plt
import math

# keep track of how long training takes
startTime = time.time()
endTime = time.time()

# set the seed and make sure tensorflow is working
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# configure the dimensions of the training images
# and mini batch size
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 100
NUM_BREEDS = 120
FLOAT_TYPE = tf.float32

MAX_LR = 0.003
MIN_LR = 0.0001
DECAY_SPEED = 2000.0

PDROP = 0.5

# fast leaky relu implementation
# https://github.com/tensorflow/tensorflow/issues/4079
def lrelu(x, leak=0.1, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)


# neural network with 3 convolutional layers, no pooling, and 2 fully connected layers (lReLU/softmax)
# convolutions are for feature learning, fully connected are for classification
# . . . . . . . . . . . .  (input data, not flattened, 3 channels)  X:  [batch, 64, 64, 3]
# @ @ @ @ @ @ @ @ @ @ @ @  (conv. layer, 6x6x3=>6 stride 1)         W1: [6, 6, 3, 6]        B1: [6] (width 6, height 6, 3 inputs, 6 outputs)
# ::::::::::::::::                                                  Y1: [batch, 64, 64, 6]
#     @ @ @ @ @ @ @ @      (conv. layer, 5x5x6=>8 stride 2)         W2: [5, 5, 6, 8]        B2: [8]
#     :::::::::                                                     Y2: [batch, 32, 32, 8]
#       @ @ @ @ @ @        (conv. layer, 4x4x8=>12 stride 2)        W3: [4, 4, 8, 12]       B3: [12]
#       :::::                                                       Y3: [batch, 16, 16, 12]
#       \x/x\x/x\x/        (fully connected LReLU layer, flat)      W4: [16*16*12, 128]     B4: [128]
#        . . . . .                                                  Y4: [batch, 128]
#        \x/x\x/x/         (fully connected softmax layer)          W5: [128, 120]          B5: [120]
#         . . . .                                                   Y:  [batch, 120]


# load the dog breed images and labels
deepDog = ddog.DeepDog(IMAGE_WIDTH, IMAGE_HEIGHT, trainingInRAM=True)

# input X: 64x64 color images [batch size, height, width, color channels]
X = tf.placeholder(FLOAT_TYPE, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
# labels for each image
Y_ = tf.placeholder(FLOAT_TYPE, [None, NUM_BREEDS])
# variable learning rate
LR = tf.placeholder(FLOAT_TYPE)
# drop out probability
pkeep = tf.placeholder(FLOAT_TYPE)

# three convolutional layers with channel outputs
A = 6
B = 8
C = 12
D = 128

# weights W1[6, 6, 3, 6], biases b[6] (6x6 patch, 3 input channels, 6 output channels)
W1 = tf.Variable(tf.truncated_normal([6, 6, 3, A], dtype=FLOAT_TYPE, stddev=0.1))
B1 = tf.Variable(tf.ones([A], dtype=FLOAT_TYPE) / 10)
# weights W2[5, 5, 6, 12], biases b[12]
W2 = tf.Variable(tf.truncated_normal([5, 5, A, B], dtype=FLOAT_TYPE, stddev=0.1))
B2 = tf.Variable(tf.ones([B], dtype=FLOAT_TYPE) / 10)
# weights W3[4, 4, 12, 18], biases b[18]
W3 = tf.Variable(tf.truncated_normal([4, 4, B, C], dtype=FLOAT_TYPE, stddev=0.1))
B3 = tf.Variable(tf.ones([C], dtype=FLOAT_TYPE) / 10)
# weights W4[16*16*18, 256], biases b[256]
W4 = tf.Variable(tf.truncated_normal([16*16*C, D], dtype=FLOAT_TYPE, stddev=0.1))
B4 = tf.Variable(tf.ones([D], dtype=FLOAT_TYPE) / 10)
# weights W5[256, 120], biases b[120]
W5 = tf.Variable(tf.truncated_normal([D, NUM_BREEDS], dtype=FLOAT_TYPE, stddev=0.1))
B5 = tf.Variable(tf.ones([NUM_BREEDS], dtype=FLOAT_TYPE) / 10)

# the model
stride = 1 # output is 64x64
Y1 = lrelu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2 # output is 32x32
Y2 = lrelu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2 # output is 16x16
Y3 = lrelu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from Y3 from the 3rd convolution to the fully connected lrelu layer
YY = tf.reshape(Y3, shape=[-1, 16*16*C])
Y4 = lrelu(tf.matmul(YY, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

# softmax layer
Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# the loss function: cross entropy
# cross entropy: -SUM(LABEL_i * log(PREDICTION_i))
# multiply by the batch size to normalize for multiple images instead of 1
# multiply by number of breeds (classes) because reduce_mean takes a mean
# and divides by the number of classes
# -------------------------------------------------------------------------
# log takes the log of each element in the array
# * multiplies elementwise
# reduce_mean sums all the elements in the array and divides by the # elems
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * BATCH_SIZE * NUM_BREEDS

# i was getting NaN with the above cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * BATCH_SIZE

# accuracy of model (0 is worst, 1 is best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, FLOAT_TYPE))

# top k accuracy
k = 5
top_k_prediction = tf.nn.in_top_k(Y, tf.argmax(Y_, 1), k)
top_k_accuracy = tf.reduce_mean(tf.cast(top_k_prediction, FLOAT_TYPE))

# training step, learning rate
# minimize the loss function
# learning rate LR is variable
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# initialize all the weights and biases
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
sess.run(init)

def training_step(i, eval_test_data, eval_train_data):

    # get the training images and labels of size BATCH_SIZE
    batch_X, batch_Y = deepDog.getNextMiniBatch(BATCH_SIZE)

    # evaluate the performance on the training data
    if eval_train_data:
        acc, cross, topk = sess.run([accuracy, cross_entropy, top_k_accuracy], 
            feed_dict={X:batch_X, Y_: batch_Y, pkeep: 1.0})

        print('Iteration ' + str(i) + ': Training Accuracy: ' + \
            str(acc) + ', Training Loss: ' + str(cross) + ', Top ' + \
            str(k) + ' Accuracy: ' + str(topk))

        training_accuracies.append((i, acc))
        training_ce.append((i, cross))

    # evaluate the performance on the test data
    if eval_test_data:
        test_X, test_Y = deepDog.getTestImagesAndLabels()
        trainingSetSize = deepDog.getTrainingSetSize()
        acc, cross, topk = sess.run([accuracy, cross_entropy, top_k_accuracy], 
            feed_dict={X:test_X, Y_: test_Y, pkeep: 1.0})

        epochNum = (i * BATCH_SIZE) // trainingSetSize 
        print('********* Epoch ' + str(epochNum) + ' *********')
        print('Iteration ' + str(i) + ': Test Accuracy: ' + \
            str(acc) + ', Test Loss: ' + str(cross) + ', Top ' + \
            str(k) + ' Accuracy: ' + str(topk))

        test_accuracies.append((i, acc))
        top_k_test_accuracies.append((i, topk))
        testing_ce.append((i, cross))

    # decay the learning rate
    learning_rate = MIN_LR + (MAX_LR - MIN_LR) * math.exp(-i/DECAY_SPEED)

    # run the training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, LR: learning_rate, pkeep: PDROP})
    global endTime
    endTime = time.time()

NUM_ITERATIONS = 10000
training_accuracies = []
test_accuracies = []
training_ce = []
testing_ce = []
top_k_test_accuracies = []

for i in range(NUM_ITERATIONS+1):
    training_step(i, i % 50 == 0, i % 10 == 0)

max_test_acc = max(test_accuracies, key=lambda z: z[1])[1]
max_top_k_acc = max(top_k_test_accuracies, key=lambda z: z[1])[1]
print('Max Test Accuracy: ' + str(max_test_acc) + ', Max Top ' + str(k) + \
    ' Accuracy: ' + str(max_top_k_acc))
print('Time Elapsed (seconds): ' + str(endTime - startTime))

plt.figure(1)
plt.subplot(211)
plt.plot(*zip(*training_accuracies), label='Training')
plt.plot(*zip(*test_accuracies), label='Test')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracies')

plt.subplot(212)
plt.plot(*zip(*training_ce), label='Training')
plt.plot(*zip(*testing_ce), label='Test')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy Loss')
plt.title('Training and Test Loss')

plt.tight_layout()
plt.legend()
plt.show()

# 10k iterations, 3 conv layers [6,6,3,6], [5,5,6,8], [4,4,8,12], 128 FC LReLU
#       max test accuracy: ~0.07, max top 5 accuracy: ~0.21
#       memorized training set, stopped learning

# 10k iterations, 3 conv layers [6,6,3,6], [5,5,6,8], [4,4,8,12], 128 FC LReLU, dropout 0.75
#       max test accuracy: ~0.08, max top 5 accuracy: ~0.22
#       memorized training set, stopped learning

# 10k iterations, 3 conv layers [6,6,3,6], [5,5,6,8], [4,4,8,12], 128 FC LReLU, dropout 0.5
#       max test accuracy: ~0.08, max top 5 accuracy: ~0.25
#       memorized training set, stopped learning