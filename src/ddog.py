import util
import json
import numpy as np
import random

class DeepDog:

    def __init__(self, imageWidth, imageHeight):
        """
        """
        self.image_width = imageWidth
        self.image_height = imageHeight

        # load the one hot encodings from file
        self.one_hot_encodings = {}
        self.loadOneHotEncodings()

        # load the test set from file
        self.test_set_images, self.test_set_labels = [], []
        self.loadTestSet()

        # load the training annotations from file and randomize the 
        # order of the training examples
        # self.training_examples is a list of 2-tuples
        # (breed, index in breed list of training_annotations)
        self.training_annotations = {}
        self.training_examples = []
        self.training_set_size = 0
        self.loadTrainingSet()

        # keep track of our place in the training examples list
        # so we can get the next mini batch
        self.current_training_index = 0


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
        for breed in self.training_annotations.keys():
            for i, _ in enumerate(self.training_annotations[breed]):
                self.training_examples.append((breed, i))

        self.training_set_size = len(self.training_examples)

        # randomize the order of the training examples
        random.shuffle(self.training_examples)
        print(self.training_examples)

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
            print(str(round(i / 120.0 * 100, 2)) + "%: Loading test images for " + breed)
            
            for annotation in testing_breeds[breed]:
                # append the image data to testImages
                self.test_set_images.append(util.getResizedImageData(annotation, 
                    self.image_width, self.image_height))

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
        pass


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


dd = DeepDog(64, 64)
im, la = dd.getTestImagesAndLabels()
print(im.shape, la.shape)
print(im)
print(la)