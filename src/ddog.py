import util
import json
import numpy as np
import random

class DeepDog:

    def __init__(self, imageWidth, imageHeight, trainingInRAM=False, classStratify=False):
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
                (not implemented yet)

            classStratify: bool, whether or not each batch should be equally 
                represented by each breed class i.e. in a batch size of 120,
                each breed would show up once in the batch
                (not implemented yet)
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
            annotation = self.training_annotations[breed][index]

            # get the image data for the training example
            batchImages.append(util.getResizedImageData(annotation, 
                self.image_width, self.image_height))

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