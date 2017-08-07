import xml.etree.ElementTree as ET
import glob
import os
import json
import csv
from PIL import Image
import numpy as np


def getAnnotationDict(annotationPath):
    """
    getAnnotationDict reads an annotation file, parses
    the XML, and returns the labeled dog breed and  
    image file dimensions.

    input:
        annotationPath: string, file path of the annotation file

    output:
        result_dict: dictionary, with filename, breed, width, height,
        and bounding box 
    """

    tree = ET.parse(annotationPath)
    root = tree.getroot()

    result_dict = {}

    result_dict['filename'] = root[1].text

    if root[1].text == "%s":
        result_dict['filename'] = annotationPath.split('\\')[-1]
    
    result_dict['breed'] = root[5][0].text

    result_dict['width'] = int(root[3][0].text)
    result_dict['height'] = int(root[3][1].text)

    result_dict['xmin'] = int(root[5][4][0].text)
    result_dict['ymin'] = int(root[5][4][1].text)
    result_dict['xmax'] = int(root[5][4][2].text)
    result_dict['ymax'] = int(root[5][4][3].text)

    return result_dict


def getAverageImageDimensions():
    file_list = []

    # for fname in glob.iglob('F:/dogs/annotation/n02085620-Chihuahua/n02085620_199/**/*', recursive=True):
    #     if os.path.isfile(fname):
    #         file_list.append(getAnnotationDict(fname))

    # print(len(file_list))
    # with open('annotation_summary.json', 'w') as fout:
    #     json.dump({'list': file_list}, fout)

    with open('annotation_summary.json', 'r') as data_file:
        file_list = json.load(data_file)['list']
    file_list_len = len(file_list)

    avg_width = sum(d['width'] for d in file_list) / float(file_list_len)
    avg_height = sum(d['height'] for d in file_list) / float(file_list_len)
    avg_xmin = sum(d['xmin'] for d in file_list) / float(file_list_len)
    avg_ymin = sum(d['ymin'] for d in file_list) / float(file_list_len)
    avg_xmax = sum(d['xmax'] for d in file_list) / float(file_list_len)
    avg_ymax = sum(d['ymax'] for d in file_list) / float(file_list_len)

    print("width: ", avg_width)
    print("height: ", avg_height)
    print("box width: ", avg_xmax - avg_xmin)
    print("box height: ", avg_ymax - avg_ymin)

    hw_tuples = [(d['filename'], d['breed'], abs(d['xmax'] - d['xmin']), abs(d['ymax'] - d['ymin'])) for d in file_list]
    with open('boxes6.csv', 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(('filename', 'breed', 'width', 'height'))
        for row in hw_tuples:
            csv_out.writerow(row)

    # wrong averages
    # width:  442.5318756073858
    # height:  385.8612244897959
    # box width:  289.04067055393585
    # box height:  309.0403304178814 

    # correct averages
    # width:  442.5318756073858
    # height:  385.8612244897959
    # box width:  289.04067055393585
    # box height:  297.6299319727891


ANNOTATION_PATH = 'F:/dogs/annotation/'
IMAGE_PATH = 'F:/dogs/images/'
BOX_FOLDER = 'boxes_'


def getImageFolderPathName(annotationDict):
    """
    getImageFolderPathName returns the folder name which contains the
    image file for the annotationDict parameter.

    'n02085620_7' => 'n02085620-Chihuahua'

    input:
        annotationDict: dictionary, contains filename of annotation.

    output:
        string, returns folder name which contains image file
        for this annotation
    """
    filename = annotationDict['filename']
    folderName = filename.split('_')[0] + '-' + annotationDict['breed']
    return folderName


def getBoxFolderPathName(annotationDict, newWidth, newHeight):
    """
    getBoxFolderPathName returns the folder name which contains the
    resized image files for an original image file.

    Given image 'n02085620_7', you can find the resized images at:
        'F:/dogs/images/n02085620-Chihuahua/boxes_64_64/'

    input:
        annotationDict: dictionary, contains filename

        newWidth: int, the new width for the image

        newHeight: int, the new height for the image

    output:
        returns a string, the folder path for the resized images
    """
    folderName = getImageFolderPathName(annotationDict)
    boxFolder = BOX_FOLDER + str(newWidth) + '_' + str(newHeight)
    return IMAGE_PATH + folderName + '/' + boxFolder + '/'


def getImageFilePathName(annotationDict, newWidth, newHeight):
    """
    getImageFilePathName returns the file path and file name for the 
    new cropped and resized image.

    Given image 'n02085620_7' resized to 64 x 64,
    it will return:
        'F:/dogs/images/n02085620-Chihuahua/boxes_64_64/n02085620_7_box_64_64.jpg'

    input:
        annotationDict: dictionary, contains the filename

        newWidth: int, the new width for the image

        newHeight: int, the new height for the image

    output:
        returns a string, which is the folder path and the file name

    """
    filename = annotationDict['filename']

    # create the file suffix i.e. '_box_64x64.jpg'
    boxFileEnding = '_box_' + str(newWidth) + '_' + str(newHeight) + '.jpg'
    
    return getBoxFolderPathName(annotationDict, newWidth, newHeight) + filename + boxFileEnding


def cropSaveBoundedBox(annotationDict, newWidth, newHeight, backgroundColor=(0,0,0)):
    """
    cropSaveBoundedBox crops the image to the pixels designated
    by its bounded box, resizes the image to the provided 
    dimensions (newWidth, newHeight), and saves the new cropped 
    and resized image to disk.  It maintains the aspect ratio
    of the original image by placing the image on a black background.

    input:
        annotationDict: dictionary, returned from getAnnotationDict(),
        contains filename, breed, and other info

        newWidth: int, new width in px for resized bounded box

        newHeight: int, new height in px for resized bounded box 

        backgroundColor: (int, int, int), color for background

    output:
        returns nothing, will throw exception if crop or save fails
    """
    
    # open the original image (the one which is not resized)
    # for instance, image file n02085620_7 is located at 'F:/dogs/images/n02085620-Chihuahua/n02085620_7.jpg'
    filename = annotationDict['filename']
    folderName = getImageFolderPathName(annotationDict)
    tempImage = Image.open(IMAGE_PATH + folderName + '/' + filename + '.jpg')

    # crop the image to the region defined by the bounding box
    croppedImage = tempImage.crop((annotationDict['xmin'], 
                                   annotationDict['ymin'],
                                   annotationDict['xmax'],
                                   annotationDict['ymax']))

    # create an empty black background size of the new image size
    background = Image.new('RGB', (newWidth, newHeight), backgroundColor)

    # keep the aspect ratio of the bounding box 
    # if the width is bigger than the height
    #   boxHeight = (boxHeight / boxWidth) * newWidth
    # if the height is bigger than the width
    #   boxWidth = (boxWidth / boxHeight) * newHeight

    boxWidth = annotationDict['xmax'] - annotationDict['xmin']
    boxHeight = annotationDict['ymax'] - annotationDict['ymin']

    if boxWidth > boxHeight:
        boxHeight = int((boxHeight * newWidth) / boxWidth)
        boxWidth = newWidth
    else:
        boxWidth = int((boxWidth * newHeight) / boxHeight)
        boxHeight = newHeight

    # resize the bounding box while keeping the aspect ratio
    resizedImage = croppedImage.resize((boxWidth, boxHeight), resample=Image.LANCZOS)

    # paste the bounding box with original aspect ratio onto black background
    background.paste(resizedImage, 
        (int((newWidth - boxWidth) / 2), int((newHeight - boxHeight) / 2)))

    # save the bounding box on black background to disk
    background.save(getImageFilePathName(annotationDict, newWidth, newHeight))


def generateAllResizedImages(newWidth, newHeight, backgroundColor=(0,0,0)):
    """ 
    generateAllResizedImages reads each annotation file in 
    /annotation/* and then creates a resized image based on the
    bounding box for that annotation and saves it to the folder
    /images/breed/boxes_newWidth_newHeight 

    input:
        newWidth: int, the width for the output image

        newHeight: int, the height for the output image

        backgroundColor: (int, int, int), color for background

    output:
        returns nothing, but saves images to the corresponding
        breed folders in /images
    """
    count = 0

    # recursively iterate through all the annotation files
    for fname in glob.iglob(ANNOTATION_PATH + '**/*', recursive=True):
        if os.path.isfile(fname):
            annotationDict = getAnnotationDict(fname)
            boxFolderPath = getBoxFolderPathName(annotationDict, newWidth, newHeight)

            # create the 'boxes_newWidth_newHeight' folder if it
            # does not already exist
            if not os.path.exists(boxFolderPath):
                os.makedirs(boxFolderPath)

            # only write a new image if we haven't come across it yet
            if not os.path.exists(getImageFilePathName(annotationDict, newWidth, newHeight)):
                # crop and save the new image file
                cropSaveBoundedBox(annotationDict, newWidth, newHeight, backgroundColor)

            count += 1
            if count % 100 == 0:
                print('Progress: ' + str(count / float(20580) * 100) + '%')
                print('Just processed ' + getImageFilePathName(annotationDict, newWidth, newHeight))

    print('Images Resized:', count)


def getResizedImageData(annotationDict, width, height):
    """
    getResizedImageData returns a numpy array of the rgb values for the resized image.
    The RGB int values are converted to a float16 (0-255 => 0.0-1.0).
    The shape of the numpy array is (height, width, 3).  

    input:
        annotationDict: dictionary, contains the file name for the original image,
                        used to get the filepath for the resized image

        width: int, the width of the resized image to retrieve

        height: int, the height of the resized image to retrieve

    output:
        returns a numpy array of shape (height, width, 3) representing
        the resized image data 
    """
    filePath = getImageFilePathName(annotationDict, width, height)
    image = Image.open(filePath)
    imageArray = np.array(image)
    # converting to float64 here increases the size per image by a factor of 8 compared
    # to the 1 byte used by 0 - 255 rgb int
    # the RAM usage is insane
    #imageArray = imageArray / 255.0 

    # converting to float 16 cuts the size by 4
    imageArray = np.array(imageArray / 255.0, dtype=np.float16)
    image.close()
    return imageArray


def generateTrainingTestLists(trainingRatio=0.7):
    """
    generateTrainingTestLists outputs three .json files containing
    the training and test splits based on the training ratio, along
    with the one hot encodings for each class (breed).  The .json
    files for the training and test splits contain dictionaries, 
    where the key is the breed, and the value for each key is a list 
    of annotation dictionaries where each annotation dicionary's 
    breed is the key.  The .json file for the one hot encodings
    is a dictionary, where the key is a breed, and the value
    is a one hot encoding (a list).

    generateTrainingTestLists reads all the annotation dictionaries
    from annotation_summary.json.

    input:
        trainingRatio: float, default is 70% training split

    output:
        writes three .json files to disk:
            training_annotations.json,
            testing_annotations.json,
            one_hot_encodings.json  
    """

    # read all the annotation dictionaries from
    # annotation_summary.json into file_list
    file_list = []
    with open('annotation_summary.json', 'r') as data_file:
        file_list = json.load(data_file)['list']

    # sort the annotations based on breed into a dictionary
    # i.e.: breedAnnotations = {'husky': [list of husky annotations]}
    breedAnnotations = {}
    for annotation in file_list:
        breed = annotation['breed']
        if breed not in breedAnnotations:
            breedAnnotations[breed] = [annotation]
        else:
            breedAnnotations[breed].append(annotation)

    # dictionaries for training splits, testing splits, and one hot encodings
    trainingAnnotations = {}
    testingAnnotations = {}
    oneHotEncodings = {}
    numberBreeds = len(breedAnnotations.keys())

    for i, breed in enumerate(breedAnnotations.keys()):
        # calculate the number of training examples to include from this breed,
        # as each breed has a different number of total annotations
        breedAnnotationsLen = len(breedAnnotations[breed])
        trainingSize = int(breedAnnotationsLen * trainingRatio)
        trainingAnnotations[breed] = breedAnnotations[breed][:trainingSize]
        testingAnnotations[breed] = breedAnnotations[breed][trainingSize:]

        # create the one hot encoding for this breed
        # the one hot encoding is a list of 0s, the size being the number of breeds
        # a 1 is inserted at the index of the ith breed for the encodings
        oneHotEncodings[breed] = ([0] * numberBreeds)
        oneHotEncodings[breed][i] = 1

    # write the three dictionaries to file
    with open('training_annotations.json', 'w') as fout:
        json.dump(trainingAnnotations, fout)

    with open('testing_annotations.json', 'w') as fout:
        json.dump(testingAnnotations, fout)

    with open('one_hot_encodings.json', 'w') as fout:
        json.dump(oneHotEncodings, fout)


#getAverageImageDimensions()
#annotationDict = getAnnotationDict('F:/dogs/annotation/n02085620-Chihuahua/n02085620_2208')
#cropSaveBoundedBox(annotationDict, 280, 291)

#generateAllResizedImages(280, 291)

#generateTrainingTestLists()

# annotationDict = getAnnotationDict('F:/dogs/annotation/n02085620-Chihuahua/n02085620_2903')
# getResizedImageData(annotationDict, 64, 64)


#generateAllResizedImages(64, 64, (0, 255, 0))
