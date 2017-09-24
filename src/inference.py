import tensorflow as tf
import util
from PIL import Image
import numpy as np
import json

inferenceImageArray = []
inferenceImage = Image.open("./saved_models/09232017/gs.jpg")
resizedImage = inferenceImage.resize((128, 128), resample=Image.LANCZOS)
resizedImageNP = np.array(resizedImage)
resizedImageNP = np.array(resizedImageNP / 255.0, dtype=np.float16)
inferenceImageArray.append(resizedImageNP)
inferenceImageArray = np.array(inferenceImageArray)
print(inferenceImageArray.shape)

sess = tf.Session()
saver = tf.train.import_meta_graph('./saved_models/09232017_2/test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./saved_models/09232017_2/'))

sess.run(tf.global_variables_initializer())

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
pkeep = graph.get_tensor_by_name('pkeep:0')

feed_dict = {X: inferenceImageArray, pkeep:1.0}

Y = graph.get_tensor_by_name('Y:0')

oneHotPrediction = sess.run(Y, feed_dict=feed_dict)
print(oneHotPrediction)
indexOfPrediction = np.argmax(oneHotPrediction)
print(indexOfPrediction)

topIndices = (np.argsort(oneHotPrediction))[0][::-1]
print(topIndices)

oneHotEncodings = {}
with open('one_hot_encodings.json') as data_file:
    oneHotEncodings = json.load(data_file)

oneHotToBreed = util.oneHotEncodingToClass(oneHotEncodings)

for i in range(5):
    breedName = oneHotToBreed[topIndices[i]]
    print("Predicted breed " + str(i+1) + ": " + breedName)