import xml.etree.ElementTree as ET

def getAnnotationDict(annotationPath):
    """
    getAnnotationDict reads an annotation file, parses
    the XML, and returns the labeled dog breed and the 
    image file dimensions.

    input:
        annotationPath: string, file path of the annotation file

    output:
        result_dict: dictionary with filename, breed, width, height,
        and bounding box 
    """

    tree = ET.parse(annotationPath)
    root = tree.getroot()

    result_dict = {}

    result_dict['filename']= root[1].text
    result_dict['breed'] = root[5][0].text

    result_dict['width'] = int(root[3][0].text)
    result_dict['height'] = int(root[3][1].text)

    result_dict['xmin'] = int(root[5][4][0].text)
    result_dict['ymin'] = int(root[5][4][1].text)
    result_dict['xmax'] = int(root[5][4][2].text)
    result_dict['ymax'] = int(root[5][4][2].text)

    return result_dict

