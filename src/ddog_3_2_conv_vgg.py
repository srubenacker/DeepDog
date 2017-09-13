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
TRAIN_BATCH_SIZE = 25
TEST_BATCH_SIZE = 50
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


# neural network inspired by the VGG-11 model (without LRN)
# input images are 128x128x3
# https://arxiv.org/pdf/1409.1556.pdf
# ------input----- 128x128x3
# |   conv3-64   |
# | 2x2 max pool |
# ---------------- 64x64x64
# |   conv3-128  |
# | 2x2 max pool |
# ---------------- 32x32x128
# |   conv3-256  |
# | 2x2 max pool |
# ---------------- 16x16x256
# |   conv3-512  |
# | 2x2 max pool |
# ---------------- 8x8x512
# |   fc-1024    |
# ----------------
# |   fc-120     |
# ----------------
# |   softmax    |


# load the dog breed images and labels
deepDog = ddog.DeepDog(IMAGE_WIDTH, IMAGE_HEIGHT, trainingInRAM=True, randomMirroing=True)

# input X: 64x64 color images [batch size, height, width, color channels]
X = tf.placeholder(FLOAT_TYPE, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
# labels for each image
Y_ = tf.placeholder(FLOAT_TYPE, [None, NUM_BREEDS])
# variable learning rate
LR = tf.placeholder(FLOAT_TYPE)
# drop out probability
pkeep = tf.placeholder(FLOAT_TYPE)

# convolutional layers

# weights W1[3, 3, 3, 64], biases b[64] (3x3 patch, 3 input channels, 64 output channels)
conv3_64_w = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=FLOAT_TYPE, stddev=0.1))
conv3_64_b = tf.Variable(tf.zeros([64], dtype=FLOAT_TYPE))

# weights W2[3, 3, 64, 128], biases b[128] (3x3 patch, 64 input channels, 128 output channels)
conv3_128_w = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=FLOAT_TYPE, stddev=0.1))
conv3_128_b = tf.Variable(tf.zeros([128], dtype=FLOAT_TYPE))

# weights W3[3, 3, 128, 256], biases b[256] (3x3 patch, 128 input channels, 256 output channels)
conv3_256_w = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=FLOAT_TYPE, stddev=0.1))
conv3_256_b = tf.Variable(tf.zeros([256], dtype=FLOAT_TYPE))

# weights W4[3, 3, 256, 512], biases b[512] (3x3 patch, 256 input channels, 512 output channels)
conv3_512_w = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=FLOAT_TYPE, stddev=0.1))
conv3_512_b = tf.Variable(tf.zeros([512], dtype=FLOAT_TYPE))

# fully connected layer (flatten)
fc_w = tf.Variable(tf.truncated_normal([8*8*512, 1024], dtype=FLOAT_TYPE, stddev=0.1))
fc_b = tf.Variable(tf.zeros([1024], dtype=FLOAT_TYPE))

# output layer weights (softmax output)
output_w = tf.Variable(tf.truncated_normal([1024, NUM_BREEDS], dtype=FLOAT_TYPE, stddev=0.1))
output_b = tf.Variable(tf.zeros([NUM_BREEDS], dtype=FLOAT_TYPE))

# the model
conv_stride = 1
pool_size = 2

# move the conv filter 1 to the right and 1 down
Y1 = lrelu(tf.nn.conv2d(X, conv3_64_w, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + conv3_64_b)

# ksize is size of pool (i.e. 2x2), strides is how to move the pool (i.e. move 2 over and 2 down)
# Y1_pool is 64x64x64 after pooling
Y1_pool = tf.nn.max_pool(Y1, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')

# Y2_pool is 32x32x128 after pooling
Y2 = lrelu(tf.nn.conv2d(Y1_pool, conv3_128_w, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + conv3_128_b)
Y2_pool = tf.nn.max_pool(Y2, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')

# Y3_pool is 16x16x256 after pooling
Y3 = lrelu(tf.nn.conv2d(Y2_pool, conv3_256_w, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + conv3_256_b)
Y3_pool = tf.nn.max_pool(Y3, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')

# Y4_pool is 8x8x512 after pooling
Y4 = lrelu(tf.nn.conv2d(Y3_pool, conv3_512_w, strides=[1, conv_stride, conv_stride, 1], padding='SAME') + conv3_512_b)
Y4_pool = tf.nn.max_pool(Y4, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')

# reshape and flatten the output from Y4_pool from the convolution to the fully connected lrelu layer
Y4_pool_flatten = tf.reshape(Y4_pool, shape=[-1, 8*8*512])
Y5 = lrelu(tf.matmul(Y4_pool_flatten, fc_w) + fc_b)
Y5d = tf.nn.dropout(Y5, pkeep)

# softmax layer
Ylogits = tf.matmul(Y5d, output_w) + output_b
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
        test_acc = correct_count / testSetSize
        test_topk = top_k_correct_count / testSetSize
        print('********* Epoch ' + str(epochNum) + ' *********')
        print('Iteration ' + str(i) + ': Test Accuracy: ' + \
            str(test_acc) + ', Test Loss: ' + str(cross) + ', Top ' + \
            str(k) + ' Accuracy: ' + str(test_topk))

        test_accuracies.append((i, test_acc))
        top_k_test_accuracies.append((i, test_topk))
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

try:
    for i in range(NUM_ITERATIONS+1):
        training_step(i, i % 50 == 0, i % 10 == 0)
except KeyboardInterrupt:
    pass

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

# 4600 iterations, 3x3 convs: 64, 128, 256, and 512 weight layers, 2x2 max pooling, 1024 FC LReLU
#       max test accuracy: 0.1871, max top 5 accuracy: 0.4432
# stopped early at 4600 because training set was memorized

# 4600 iterations, 3x3 convs: 64, 128, 256, and 512 weight layers, 2x2 max pooling, 1024 FC LReLU,
# random mirroring
#       max test accuracy: 0.1971, max top 5 accuracy: 0.45
# stopped early at 4600 because training set was memorized