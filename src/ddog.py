import util
import json
import numpy as np
import random
import tensorflow as tf

class DeepDog:
    """
    The DeepDog class loads the training and test set images from
    disk into RAM, and provides functions to get the test set
    and mini batches of the training set. 
    """

    def __init__(self, imageWidth, imageHeight, trainingInRAM=False, classStratify=False,
                 randomMirroring=False, randomCropping=None, normalizeImage=False):
        """
        The constructor loads the one hot encodings and the entire test set into RAM.
        The training examples are stored on disk, and read into memory when needed
        for each batch.  

        input:
            imageWidth: int, width of each image

            imageHeight: int, height of each image

            trainingInRAM: bool, whether or not to load the entire training set
                           into RAM on initialization.  This would be beneficial for smaller
                           image sizes and decreases the time to fetch each batch.

            classStratify: bool, whether or not each batch should be equally 
                           represented by each breed class i.e. in a batch size of 120,
                           each breed would show up once in the batch
                           (not implemented yet)

            randomMirroring: bool, whether or not to randomly mirror individual 
                             training images returned by getNextMiniBatch()

            randomCropping: tuple, (cropWidth, cropHeight), cropWidth and cropHeight
                            are the dimensions of the cropped image returned by
                            getNextMiniBatch()

            normalizeImage: bool, whether or not to scale the images returned
                            by getNextMiniBatch() to have 0 mean and unit standard
                            deviation
        """
        self.MIRROR_PROBABILITY = 0.5
        self.randomMirroring = randomMirroring
        self.randomCropping = randomCropping
        if self.randomCropping is not None:
            self.cropWidth = self.randomCropping[0]
            self.cropHeight = self.randomCropping[1]
        self.normalizeImage = normalizeImage

        self.image_width = imageWidth
        self.image_height = imageHeight
        self.training_in_RAM = trainingInRAM

        # load the one hot encodings from file
        self.one_hot_encodings = {}
        self.loadOneHotEncodings()
        self.numberBreeds = float(len(self.one_hot_encodings.keys()))

        # load the test set from file
        self.test_set_images, self.test_set_labels = [], []
        self.loadTestSet()

        # load the training annotations from file and randomize the 
        # order of the training examples
        # self.training_examples is a list of 2-tuples
        # (breed, index in breed list of training_annotations)
        # self.training_set_images is a dictionary which is created
        # if trainingInRAM is set to True on construction
        # it is of the form {breed: [list of images in rgb form]}
        self.training_annotations = {}
        self.training_set_images = {}
        self.training_examples = []
        self.training_set_size = 0
        self.loadTrainingSet()

        # keep track of our place in the training examples list
        # so we can get the next mini batch
        self.current_index = 0


    ####################################################
    ################ Private Methods ###################
    ####################################################


    def loadOneHotEncodings(self):
        """
        loadOneHotEncodings reads the one hot encodings for each
        breed and saves them to a member dictionary.

        input: none

        output: (doesn't return, saves to member variable)
            self.one_hot_encodings: dictionary, {'breed': [1, 0, 0]}
        """
        with open('one_hot_encodings.json', 'r') as data_file:
            self.one_hot_encodings = json.load(data_file)


    def loadTrainingSet(self):
        """
        loadTrainingSet reads the training_annotations.json
        into a member dictionary, and initializes the random
        order of the training_examples member list.

        input: none

        output: (doesn't return, saves to member variables)
            self.training_annotations: dictionary, {'breed': [list of annotations]}

            self.training_examples: list of 2-tuples
                [(breed, index into list of self.training_annotations), ...]
        """
        print("Initializing training set order...\n")

        # load the training_annotations
        with open('training_annotations.json', 'r') as data_file:
            self.training_annotations = json.load(data_file)

        # create the list of 2-tuples of training examples (breed, index)
        for j, breed in enumerate(self.training_annotations.keys()):
            if self.training_in_RAM:
                print(str(round(j / self.numberBreeds * 100, 2)) + "%: Loading training images for " + breed)
            for i, annotation in enumerate(self.training_annotations[breed]):
                self.training_examples.append((breed, i))
                # if training_in_RAM is True, load the image from disk
                if self.training_in_RAM:
                    currentImage = util.getResizedImageData(annotation, self.image_width, self.image_height)
                    if breed not in self.training_set_images:
                        self.training_set_images[breed] = [currentImage]
                    else:
                        self.training_set_images[breed].append(currentImage)

        self.training_set_size = len(self.training_examples)

        # randomize the order of the training examples
        random.shuffle(self.training_examples)

        print("Finished initializing training set order...\n")


    def loadTestSet(self):
        """
        loadTestSet reads the test set images and labels from file
        and saves them into two lists in RAM.  

        input: none

        output: (saves to member lists, doesn't return)
            testImages: numpy array [testSetSize x [imageWidth x imageHeight x 3]]

            testLabels: numpy array [testSetSize x [numImageClasses]] 
        """
        print("Loading test set...\n")

        testing_breeds = {}
        with open('testing_annotations.json', 'r') as data_file:
            testing_breeds = json.load(data_file)

        for i, breed in enumerate(testing_breeds.keys()):
            print(str(round(i / self.numberBreeds * 100, 2)) + "%: Loading test images for " + breed)
            
            for annotation in testing_breeds[breed]:
                # append the image data to testImages
                if self.randomCropping is None:
                    self.test_set_images.append(util.getResizedImageData(annotation, 
                        self.image_width, self.image_height))
                else:
                    self.test_set_images.append(util.getResizedImageData(annotation, 
                        self.cropWidth, self.cropHeight))

                # append the image label's one hot encoding to testLabels
                self.test_set_labels.append(self.one_hot_encodings[annotation['breed']])

        # convert python lists to numpy arrays
        self.test_set_images = np.array(self.test_set_images)
        self.test_set_labels = np.array(self.test_set_labels)

        print("Finished loading test set.....\n")


    ####################################################
    ################ Public Interface ##################
    ####################################################


    def getNextMiniBatch(self, batchSize):
        """
        getNextMiniBatch returns a 2-tuple of (batchImages, batchLabels).
        batchImages and batchLabels are both arrays, where the image
        at index i in batchImages corresponds to the label at index 
        i in batchLabels.  The batch images and labels are from
        the training set.

        input: 
            batchSize: int, number of images and labels to include
            in the mini batch returned by getNextMiniBatch

        output:
            batchImages: numpy array [batchSize x [imageWidth x imageHeight x 3]]

            batchLabels: numpy array [batchSize x [numImageClasses]]
        """
        batchImages = []
        batchLabels = []

        # if we have reached the end of the training examples, 
        # reshuffle the training examples and start from the 
        # beginning of the list
        # in the event that the number of training examples
        # is not evenly divisable by the batchSize,
        # some training examples will be skipped during this reshuffling
        # i trade this off for decreased code complexity
        if self.current_index + batchSize > self.training_set_size:
            self.current_index = 0
            random.shuffle(self.training_examples)

        # for each training example annotation, load the resized image and
        # get the one hot encoding of the label
        for breed, index in self.training_examples[self.current_index:self.current_index+batchSize]:
            # placeholder image variable
            imageToAppend = None

            # if the training data is already in RAM, read it from self.training_set_images
            # otherwise, fetch the image from disk
            if self.training_in_RAM:
                imageToAppend = self.training_set_images[breed][index]
            else:
                annotation = self.training_annotations[breed][index]

                # get the image data for the training example
                imageToAppend = util.getResizedImageData(annotation, 
                    self.image_width, self.image_height)

            # mirror the image if the random number is less than the probability
            if self.randomMirroring and random.random() < self.MIRROR_PROBABILITY:
                imageToAppend = np.fliplr(imageToAppend)

            # randomly crop the image
            if self.randomCropping is not None:
                widthDiff = self.image_width - self.cropWidth
                heightDiff = self.image_height - self.cropHeight

                widthOffset = int(random.random() * widthDiff)
                heightOffset = int(random.random() * heightDiff)

                imageToAppend = imageToAppend[widthOffset:widthOffset+self.cropWidth, 
                                              heightOffset:heightOffset+self.cropHeight, 
                                              :]

            # # normalize the image to 0 mean and unit standard deviation
            # if self.normalizeImage:
            #     imageToAppend = tf.image.per_image_standardization(imageToAppend)

            # finally append the image
            batchImages.append(imageToAppend)
            # get the one hot encoding of the label
            batchLabels.append(self.one_hot_encodings[breed])

        self.current_index += batchSize

        return np.array(batchImages), np.array(batchLabels)


    def getTestImagesAndLabels(self):
        """
        getTestImagesAndLabels returns a 2-tuple of (testImages, testLabels).
        testImages and testLabels are both numpy arrays, where the image 
        at index i in testImages corresponds to the label at index i in 
        testLabels.  

        input: none

        output:
            testImages: numpy array [testSetSize x [imageWidth x imageHeight x 3]]

            testLabels: numpy array [testSetSize x [numImageClasses]] 
        """
        return self.test_set_images, self.test_set_labels


    def getTrainingSetSize(self):
        """
        getTraininSetSize returns the size of the training set.  This
        function is useful when computing the progress inside an epoch.

        input: none

        output:
            trainingSetSize: int, number of examples in the training set
        """
        return self.training_set_size


def main():
    dd = DeepDog(64, 64)
    im, la = dd.getNextMiniBatch(100)
    print(im.shape, la.shape)
    print(im)
    print(la)


if __name__ == "__main__":
    main()