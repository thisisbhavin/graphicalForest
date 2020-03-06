import os
import pandas as pd
from google.cloud import vision
from google.cloud.vision import types
from re import sub
import numpy as np

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './capfirst-service-role.json'

def load_image(img_filename):
    with open(img_filename, 'rb') as image_file:
        content = bytearray(image_file.read())
        image_file.close()
    return content

def ocr_processing(path):
    data = load_image(path)
    image = types.Image(content=bytes(data))
    client = vision.ImageAnnotatorClient()
    textdetection_response = client.document_text_detection(image=image)
    return textdetection_response

def get_eng_words_bb(gvision_ocr_text_description):
    """Gives list of coordinates of the bounding box of each english word from gvision text.
    """
    text_list = gvision_ocr_text_description.text_annotations[1:]
    coord_list = []
    word_list = []
    for text in text_list:
        # removing non ascii characters to get eng text and "/" to mimic preprocessing function
        plain_eng_text = text.description.encode('ascii', 'ignore').decode('utf-8').upper()
        plain_eng_text = sub(r'[^A-Za-z0-9\n\s]+', '', plain_eng_text)
        plain_eng_text = sub(r"[\s]+"," ", plain_eng_text.strip())
        if plain_eng_text:
            word_list.append(plain_eng_text)
            coord_list.append([[v.x,v.y] for v in text.bounding_poly.vertices])
    coord_array = np.array(coord_list)

    return coord_array, word_list

def ocr_using_google_api(image_path):
    '''
    This function uses Google Vision API for Text Detection

    Args :
        image_path : Input image path

    Returns:
        pd.DataFrame having coordinates of each text box along with detected
        text inside it.
    '''
    textdetection_response = ocr_processing(image_path)
    coord_array, word_list = get_eng_words_bb(textdetection_response)
    coord_object = pd.DataFrame(coord_array[:,(0,2),:].reshape(-1,4), columns=['xmin','ymin','xmax','ymax'])
    coord_object['Object'] = word_list

    return coord_object

if __name__ == '__main__':
    coord_object = ocr_using_google_api('data/deskew.jpg')
    coord_object.to_csv('ab.csv')
    print(coord_object)
    