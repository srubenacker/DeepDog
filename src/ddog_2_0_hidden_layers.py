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
BATCH_SIZE = 100
NUM_BREEDS = 120
FLOAT_TYPE = tf.float32
MAX_LR = 0.003
MIN_LR = 0.0001
DECAY_SPEED = 2000.0
PDROP = 0.75

# fast leaky relu implementation
# https://github.com/tensorflow/tensorflow/issues/4079
def lrelu(x, leak=0.1, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)


# neural network with 1 input layer, 3 hidden layers, and 1 output layer
# . . . . . . . . . . . . . .    -- input layer, flattened rgb pixels              X  [batch, 12288]
# \x/\x/\x/\x/\x/\x/\x/\x/\x/ D  -- fully connected hidden layer (lRelu/dropout)   W1 [12288, 512] B1[512]
#  .  .  .  .  .  .  .  .  .                                                       Y1 [batch, 512]
#  \x/\x/\x/\x/\x/\x/\x/\x/   D  -- fully connected hidden layer (lRelu/dropout)   W2 [512,   256] B2[256]
#   .  .  .  .  .  .  .  .                                                         Y2 [batch, 256]
#   \x/\x/\x/\x/\x/\x/\x/     D  -- fully connected hidden layer (lRelu/dropout)   W3 [256,   128] B3[128]
#    .  .  .  .  .  .  .                                                           Y3 [batch, 128]
#    \x/\x/\x/\x/\x/\x/          -- fully connected output layer (softmax)         W4 [128,   120] B4[120]
#     .  .  .  .  .  .                                                             Y  [batch, 120]

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

IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * 3 # 12,288 = 64*64*3
FIRST_LAYER = 256
SECOND_LAYER = 128
#THIRD_LAYER = 128
FOURTH_LAYER = NUM_BREEDS

# weights W1[12288, 512], biases b[512]
W1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, FIRST_LAYER], dtype=FLOAT_TYPE, stddev=0.1))
B1 = tf.Variable(tf.ones([FIRST_LAYER], dtype=FLOAT_TYPE) / 10)
# weights W2[512, 256], biases b[256]
W2 = tf.Variable(tf.truncated_normal([FIRST_LAYER, SECOND_LAYER], dtype=FLOAT_TYPE, stddev=0.1))
B2 = tf.Variable(tf.ones([SECOND_LAYER], dtype=FLOAT_TYPE) / 10)
# weights W3[256, 128], biases b[128]
W3 = tf.Variable(tf.truncated_normal([SECOND_LAYER, FOURTH_LAYER], dtype=FLOAT_TYPE, stddev=0.1))
B3 = tf.Variable(tf.ones([FOURTH_LAYER], dtype=FLOAT_TYPE) / 10)
# weights W4[128, 120], biases b[120]
# W4 = tf.Variable(tf.truncated_normal([THIRD_LAYER, FOURTH_LAYER], dtype=FLOAT_TYPE, stddev=0.1))
# B4 = tf.Variable(tf.zeros([FOURTH_LAYER], dtype=FLOAT_TYPE))

# flatten the image into a single line of pixels
XX = tf.reshape(X, [-1, IMAGE_PIXELS])

# the model
Y1 = lrelu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = lrelu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

#Y3 = lrelu(tf.matmul(Y2d, W3) + B3)
#Y3d = tf.nn.dropout(Y3, pkeep)

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
config.gpu_options.per_process_gpu_memory_fraction = 0.5
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

# 10k iterations, single sigmoid hidden layer, W1=512
#       max test accuracy: 0.0307, max top 5 accuracy: 0.1259
# 10k iterations, single Relu hidden layer, W1=512
#       max test accuracy: 0.0122, max top 5 accuracy: 0.0562
# 10k iterations, single lRelu hidden layer, W1=512
#       max test accuracy: 0.0668, max top 5 accuracy: 0.1836
# 10k iterations, single lRelu hidden layer, W1=256
#       max test accuracy: 0.0612, max top 5 accuracy: 0.1886

# 10k iterations, two sigmoid hidden layers, W1=512, W2=256
#       max test accuracy: 0.0410, max top 5 accuracy: 0.1674
# 10k iterations, two relu hidden layers, W1=512, W2=256
#       max test accuracy: 0.0545, max top 5 accuracy: 0.2
# 10k iterations, two lrelu hidden layers, W1=512, W2=256
#       max test accuracy: 0.0771, max top 5 accuracy: 0.2332
# 10k iterations, two lrelu hidden layers, W1=256, W2=128
#       max test accuracy: 0.0805, max top 5 accuracy: 0.2402
# 10k iterations, two lrelu hidden layers, W1=128, W2=128
#       max test accuracy: 0.0819, max top 5 accuracy: 0.2410

# 10k iterations, two lrelu hidden layers, W1=128, W2=128, dropout
#       max test accuracy: 0.0909, max top 5 accuracy: 0.2511
# 10k iterations, two lrelu hidden layers, W1=256, W2=128, dropout
#       max test accuracy: 0.0930, max top 5 accuracy: 0.2657

# 10k iterations, three lrelu hidden layers, W1=512, W2=256, W3=128, dropout
#       max test accuracy: 0.0903, max top 5 accuracy: 0.2702
# 10k iterations, three lrelu hidden layers, W1=256, W2=128, W3=128, dropout
#       max test accuracy: 0.0876, max top 5 accuracy: 0.2578

# 10k iterations, two lrelu hidden layers, W1=256, W2=128, dropout, 128x128 images
#       max test accuracy: 0.0986, max top 5 accuracy: 0.2826