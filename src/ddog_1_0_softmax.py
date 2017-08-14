import tensorflow as tf
import time
import ddog

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
BATCH_SIZE = 50
NUM_BREEDS = 120
FLOAT_TYPE = tf.float32

# neural network with 1 input layer, and 1 fully connected output layer
# 
# x_1 x_2 x_3 ... x_12288 (input layer, flattened rgb pixels) X: [batch size, 12288]
#  \   /   \  ...    /                                        64 * 64 * 3 = 12,288
# <fully connected weights layer, biases>                     W: [12288, 120], b: [120]
#  \.../  ...  \.../                                          1,474,560 weights
# breed_0 ... breed_119 (output layer, softmax)               Y: [batch size, 120]
#                                                             120 dog breeds

# The model
# 
# Y = softmax(X * W + b)
# X: matrix of 100 color images of 64x64 pixels, flattened
# W: weight matrix with 1,474,560 weights, 120 columns
# X * W: [100, 120]
# b: bias vector with 120 dimensions
# +: add with broadcasting: adds the vector to each line of the matrix (numpy)
# Y: output matrix with 100 lines and 120 columns

# load the dog breed images and labels
deepDog = ddog.DeepDog(IMAGE_WIDTH, IMAGE_HEIGHT)

# input X: 64x64 color images [batch size, height, width, color channels]
X = tf.placeholder(FLOAT_TYPE, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
# labels for each image
Y_ = tf.placeholder(FLOAT_TYPE, [None, NUM_BREEDS])
# weights W[12288, 120] 12,288 = 64*64*3
W = tf.Variable(tf.zeros([IMAGE_HEIGHT * IMAGE_WIDTH * 3, NUM_BREEDS], dtype=FLOAT_TYPE))
# biases b[120]
b = tf.Variable(tf.zeros([NUM_BREEDS], dtype=FLOAT_TYPE))

# flatten the image into a single line of pixels
XX = tf.reshape(X, [-1, IMAGE_HEIGHT * IMAGE_WIDTH * 3])

# the model
Ylogits = tf.matmul(XX, W) + b
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

# training step, learning rate = 0.005
# minimize the loss function
LEARNING_RATE = 0.005
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# initialize all the weights and biases
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, eval_test_data, eval_train_data, max_test_acc):

    # get the training images and labels of size BATCH_SIZE
    batch_X, batch_Y = deepDog.getNextMiniBatch(BATCH_SIZE)

    print('Iteration ' + str(i))
    # print('Batch X: ' + str(batch_X))
    # print('Batch X Shape: ' + str(batch_X.shape))
    # print('Batch Y: ' + str(batch_Y))
    # print('Batch Y Shape: ' + str(batch_Y.shape))

    # evaluate the performance on the training data
    if eval_train_data:
        acc, cross = sess.run([accuracy, cross_entropy], 
            feed_dict={X:batch_X, Y_: batch_Y})

        print('Iteration ' + str(i) + ': Training Accuracy: ' + \
            str(acc) + ': Training Loss: ' + str(cross))

    # evaluate the performance on the test data
    if eval_test_data:
        test_X, test_Y = deepDog.getTestImagesAndLabels()
        trainingSetSize = deepDog.getTrainingSetSize()
        acc, cross = sess.run([accuracy, cross_entropy], 
            feed_dict={X:test_X, Y_: test_Y})

        epochNum = (i * BATCH_SIZE) // trainingSetSize 
        print('********* Epoch ' + str(epochNum) + ' *********')
        print('Iteration ' + str(i) + ': Test Accuracy: ' + \
            str(acc) + ': Test Loss: ' + str(cross))
        
        if acc > max_test_acc:
            max_test_acc = acc

    # run the training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
    endTime = time.time()

NUM_ITERATIONS = 2001
max_test_acc = 0
for i in range(NUM_ITERATIONS):
    training_step(i, i % 50 == 0, i % 10 == 0, max_test_acc)

print('Max Test Accuracy: ' + str(max_test_acc))
print('Time Elapsed (seconds): ' + str(endTime - startTime))










