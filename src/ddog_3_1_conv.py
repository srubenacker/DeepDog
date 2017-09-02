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
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 500
NUM_BREEDS = 120
FLOAT_TYPE = tf.float32

MAX_LR = 0.003
MIN_LR = 0.0001
DECAY_SPEED = 2000.0

KEEP_PROB = 0.5

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
# @ @ @ @ @ @ @ @ @ @ @ @  (conv. layer, 6x6x3=>12 stride 1)        W1: [6, 6, 3, 12]        B1: [6] (width 6, height 6, 3 inputs, 12 outputs)
#       < max pool >       (max pool, 2x2)                          
#       \x/x\x/x\x/                                                 Y1: [batch, 32, 32, 12]
#                                                                   W2: [32*32*12, 128]      B2: [128]
#                                                                   Y2: [batch, 128] 
#                                                                   W3: [128, 120]           B3: [120]
#                                                                   Y:  [batch, 120]
# Conv -> LReLU -> Pool -> Dense LReLU -> Softmax


# load the dog breed images and labels
deepDog = ddog.DeepDog(IMAGE_WIDTH, IMAGE_HEIGHT, trainingInRAM=True)

# input X: 64x64 color images [batch size, height, width, color channels]
X = tf.placeholder(FLOAT_TYPE, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
# labels for each image
Y_ = tf.placeholder(FLOAT_TYPE, [None, NUM_BREEDS])
# variable learning rate
LR = tf.placeholder(FLOAT_TYPE)
# drop out probability
pkeep = tf.placeholder(FLOAT_TYPE)

# three convolutional layers with channel outputs
A = 8
D = 32

# weights W1[5, 5, 3, 8], biases b[12] (6x6 patch, 3 input channels, 12 output channels)
W1 = tf.Variable(tf.truncated_normal([3, 3, 3, A], dtype=FLOAT_TYPE, stddev=0.1))
B1 = tf.Variable(tf.ones([A], dtype=FLOAT_TYPE) / 10)

# fully connected layer (flatten)
W2 = tf.Variable(tf.truncated_normal([64*64*A, D], dtype=FLOAT_TYPE, stddev=0.1))
B2 = tf.Variable(tf.ones([D], dtype=FLOAT_TYPE) / 10)

W3 = tf.Variable(tf.truncated_normal([D, NUM_BREEDS], dtype=FLOAT_TYPE, stddev=0.1))
B3 = tf.Variable(tf.ones([NUM_BREEDS], dtype=FLOAT_TYPE) / 10)

# the model
stride = 1
pool = 2
Y1 = lrelu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
# ksize is size of pool (i.e. 2x2), strides is how to move the pool (i.e. move 2 over and 2 down)
# Y1P is 32x32 after pooling
Y1P = tf.nn.max_pool(Y1, ksize=[1, pool, pool, 1], strides=[1, pool, pool, 1], padding='SAME')

# reshape the output from Y1P from the convolution to the fully connected lrelu layer
Y1P_RESHAPE = tf.reshape(Y1P, shape=[-1, 64*64*A])
Y2 = lrelu(tf.matmul(Y1P_RESHAPE, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

# softmax layer
Ylogits = tf.matmul(Y2d, W3) + B3
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
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * TRAIN_BATCH_SIZE * NUM_BREEDS

# i was getting NaN with the above cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * TRAIN_BATCH_SIZE

# accuracy of model (0 is worst, 1 is best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, FLOAT_TYPE))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction, tf.int16))

# top k accuracy
k = 5
top_k_prediction = tf.nn.in_top_k(Y, tf.argmax(Y_, 1), k)
top_k_accuracy = tf.reduce_mean(tf.cast(top_k_prediction, FLOAT_TYPE))
top_k_accuracy_count = tf.reduce_sum(tf.cast(top_k_prediction, tf.int16))

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

    # get the training images and labels of size TRAIN_BATCH_SIZE
    batch_X, batch_Y = deepDog.getNextMiniBatch(TRAIN_BATCH_SIZE)

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

        correct_count = 0
        top_k_correct_count = 0
        testSetSize = test_X.shape[0]

        j = 0
        while j < testSetSize:
            start = j 
            end = j + TEST_BATCH_SIZE
            cross, acc_count, topk_count = sess.run([cross_entropy, accuracy_count, top_k_accuracy_count], 
                                                                    feed_dict={X:test_X[start:end], 
                                                                               Y_: test_Y[start:end], 
                                                                               pkeep: 1.0})
            correct_count += acc_count
            top_k_correct_count += topk_count
            j += TEST_BATCH_SIZE

        epochNum = (i * TRAIN_BATCH_SIZE) // trainingSetSize 
        print('********* Epoch ' + str(epochNum) + ' *********')
        print('Iteration ' + str(i) + ': Test Accuracy: ' + \
            str(correct_count / testSetSize) + ', Test Loss: ' + str(cross) + ', Top ' + \
            str(k) + ' Accuracy: ' + str(top_k_correct_count / testSetSize))

        test_accuracies.append((i, acc))
        top_k_test_accuracies.append((i, topk))
        testing_ce.append((i, cross))

    # decay the learning rate
    learning_rate = MIN_LR + (MAX_LR - MIN_LR) * math.exp(-i/DECAY_SPEED)

    # run the training step
    # KEEP_PROB is 1.0 when evaluating perf on the training and test sets, and 0.5
    # when training
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, LR: learning_rate, pkeep: KEEP_PROB})
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

# 10k iterations, 1 conv layer [5,5,3,8], 2x2 max pooling, 32 FC LReLU
#       max test accuracy: 0.0887, max top 5 accuracy: 0.2461

# 10k iterations, 1 conv layer [3,3,3,8], 2x2 max pooling, 32 FC LReLU, 128x128 images
#       max test accuracy: 0.----, max top 5 accuracy: 0.----