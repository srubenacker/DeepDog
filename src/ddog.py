import util
import json


class DeepDog:

    def __init__(self, imageWidth, imageHeight):
        """
        """
        self.image_width = imageWidth
        self.image_height = imageHeight

        self.one_hot_encodings = self.loadOneHotEncodings()
        self.test_set_images, self.test_set_labels = [], []
        self.loadTestSet()


    ####################################################
    ################ Private Methods ###################
    ####################################################


    def loadOneHotEncodings(self):
        """
        loadOneHotEncodings reads the one hot encodings for each
        breed and saves them to a member dictionary.

        input: none

        output:
            oneHotEncodingsDictionary: dictionary, {'breed': [1, 0, 0]}
        """
        oneHotEncodings = {}
        with open('one_hot_encodings.json', 'r') as data_file:
            oneHotEncodings = json.load(data_file)
        return oneHotEncodings


    def loadTestSet(self):
        """
        loadTestSet reads the test set images and labels from file
        and saves them into two lists in RAM.  

        input: none

        output: (saves to member lists, doesn't return)
            testImages: numpy array [testSetSize x [imageWidth x imageHeight]]

            testLabels: numpy array [testSetSize x [numImageClasses]] 
        """
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
            batchImages: numpy array [batchSize x [imageWidth x imageHeight]]

            batchLabels: numpy array [batchSize x [numImageClasses]]
        """
        pass


    def getTestImagesAndLabels(self):
        """
        getTestImagesAndLabels returns a 2-tuple of (testImages, testLabels).
        testImages and testLabels are both arrays, where the image 
        at index i in testImages corresponds to the label at index i in 
        testLabels.  

        input: none

        output:
            testImages: numpy array [testSetSize x [imageWidth x imageHeight]]

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
        pass


dd = DeepDog(280, 291)
im, la = dd.getTestImagesAndLabels()
print(len(im), len(la))