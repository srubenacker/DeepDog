import xml.etree.ElementTree as ET
import glob
import os
import json
import csv
from PIL import Image

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

ANNOTATION_PATH = 'F:/dogs/annotation/'
IMAGE_PATH = 'F:/dogs/images/'
BOX_FOLDER = 'boxes_'

def cropSaveBoundedBox(annotationDict, savePath, newWidth, newHeight):
    """
    cropSaveBoundedBox crops the image to the pixels designated
    by its bounded box, resizes the image to the provided 
    dimensions (newWidth, newHeight), and saves the new cropped 
    and resized image to disk.  It maintains the aspect ratio
    of the original image by placing the image on a black background.

    input:
        annotationDict: dictionary, returned from getAnnotationDict(),
        contains filename, breed, and other info

        savePath: string, path to save the image to

        newWidth: int, new width in px for resized bounded box

        newHeight: int, new height in px for resized bounded box 

    output:
        returns nothing, will throw exception if crop or save fails
    """
    
    filename = annotationDict['filename']
    folderName = getImageFolderPathName(annotationDict)
    tempImage = Image.open(IMAGE_PATH + folderName + '/' + filename + '.jpg')

    # crop the image to the region defined by the bounding box
    croppedImage = tempImage.crop((annotationDict['xmin'], 
                                   annotationDict['ymin'],
                                   annotationDict['xmax'],
                                   annotationDict['ymax']))

    # create an empty black background size of the new image size
    blackBackground = Image.new('RGB', (newWidth, newHeight), (0, 0, 0))

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
    resizedImage = croppedImage.resize((boxWidth, boxHeight), 
                                       resample=Image.LANCZOS)

    # paste the bounding box with original aspect ratio onto black background
    blackBackground.paste(resizedImage, 
        (int((newWidth - boxWidth) / 2), int((newHeight - boxHeight) / 2)))

    # save the bounding box on black background to disk
    boxFileEnding = '_box_' + str(newWidth) + '_' + str(newHeight) + '.jpg'
    blackBackground.save(savePath + '/' + filename + boxFileEnding)


def generateAllResizedImages(newWidth, newHeight):
    """ 
    generateAllResizedImages reads each annotation file in 
    /annotation/* and then creates a resized image based on the
    bounding box for that annotation and saves it to the folder
    /images/breed/boxes_newWidth_newHeight 

    input:
        newWidth: int, the width for the output image

        newHeight: int, the height for the output image

    output:
        returns nothing, but saves images to the corresponding
        breed folders in /images
    """
    count = 0

    # recursively iterate through all the annotation files
    for fname in glob.iglob(ANNOTATION_PATH + '**/*', recursive=True):
        if os.path.isfile(fname):
            annotationDict = getAnnotationDict(fname)

            # create the path ending in 'boxes_newWidth_newHeight'
            boxFolderPath = IMAGE_PATH + getImageFolderPathName(annotationDict) + '/' + BOX_FOLDER + \
                            str(newWidth) + '_' + str(newHeight)

            # create the 'boxes_newWidth_newHeight' folder if it
            # does not already exist
            if not os.path.exists(boxFolderPath):
                os.makedirs(boxFolderPath)

            # crop and save the new image file
            cropSaveBoundedBox(annotationDict, boxFolderPath, newWidth, newHeight)
            count += 1

    print('Images Resized:', count)


#getAverageImageDimensions()
#annotationDict = getAnnotationDict('F:/dogs/annotation/n02085620-Chihuahua/n02085620_2208')
#cropSaveBoundedBox(annotationDict, 280, 291)

generateAllResizedImages(280, 291)



