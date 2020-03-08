import os
import pandas as pd
from google.cloud import vision
from google.cloud.vision import types
from re import sub, split
import numpy as np
import sys
import cv2
from glob import glob
np.set_printoptions(threshold=sys.maxsize)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './capfirst-service-role.json'

def load_image(img_filename):
    with open(img_filename, 'rb') as image_file:
        content = bytearray(image_file.read())
        image_file.close()
    return content

def ocr_processing(data):
    # data = load_image(path)
    image = types.Image(content=bytes(data))
    client = vision.ImageAnnotatorClient()
    textdetection_response = client.text_detection(image=image)
    return textdetection_response

def get_eng_words_bb(gvision_ocr_text_description):
    """Gives list of coordinates of the bounding box of each english word from gvision text.
    """
    text_list = gvision_ocr_text_description.text_annotations[1:]
    coord_list = []
    word_list = []
    for text in text_list:
        # removing non ascii characters to get eng text and "/" to mimic preprocessing function
        plain_eng_text = text.description.encode('ascii', 'ignore').decode('utf-8')
        plain_eng_text = sub(r'[^A-Za-z0-9\.\,\/\:\-\n\s]+', '', plain_eng_text)
        plain_eng_text = sub(r"[\s]+"," ", plain_eng_text.strip())
        if plain_eng_text:
            word_list.append(plain_eng_text)
            coord_list.append([[v.x,v.y] for v in text.bounding_poly.vertices])
    coord_array = np.array(coord_list)

    return coord_array, word_list

def reject_outliers(data, m=2):
    if np.unique(data).size == 1:
        return data[0:1]
    return data[abs(data - np.mean(data)) < (m * np.std(data))]

def get_eng_chars_bb(textdetection_response): 
    coord_list = []
    char_list = []
    
    for page in textdetection_response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        plain_eng_text = symbol.text.encode('ascii', 'ignore').decode('utf-8')
                        plain_eng_text = sub(r'[^A-Za-z0-9\.\,\/\:\-\n\s]+', '', plain_eng_text).strip()

                        if plain_eng_text:
                            char_list.append(plain_eng_text)
                            coord_list.append([[v.x,v.y] for v in symbol.bounding_box.vertices])
                            
    coord_array = np.array(coord_list)
    return coord_array, char_list


def get_sentence_bb(cleaned_sentence_list, coord_array):
    """generate word bb from characters"""
    start = 0
    sentence_bb = []
    for sentence in cleaned_sentence_list:
        word_list = list(sentence)
        words_in_sentence = len(word_list)
        if words_in_sentence==1:
            sentence_bb_coord = coord_array[start]
            sentence_bb.append(sentence_bb_coord)

        else:
            coords = coord_array[start:start+words_in_sentence]
            sentence_bb_coord=[[coords[0,0,0], coords[0,0,1]],
                               [coords[-1,1,0], coords[-1,1,1]],
                               [coords[-1,2,0], coords[-1,2,1]],
                               [coords[0,3,0], coords[0,3,1]]]
            sentence_bb.append(sentence_bb_coord)
        start += words_in_sentence

    return np.array(sentence_bb)

def preprocess(text_description):
    text_description = text_description.encode('ascii', 'ignore').decode('utf-8')
    text_description = sub(r'[^A-Za-z0-9\.\,\/\:\-\n\s]+', '', text_description)
    sentence_list = split(r'\n', text_description)
    cleaned_sentence_list = [sub(r"[\s]+"," ", line.strip()) for line in sentence_list if line.strip()] # removing blank
    cleaned_sentence_list_wo_space_list = [sub(r"[\s]+","", line.strip()) for line in cleaned_sentence_list]
    return cleaned_sentence_list, cleaned_sentence_list_wo_space_list

def get_sentence_bb1(cleaned_sentence_list, coord_array):
    """generate sentences bb from words"""
    start = 0
    sentence_bb = []
    for sentence in cleaned_sentence_list:
        word_list = sentence.split(" ")
        word_list = [word for word in word_list if word.strip()]
        words_in_sentence = len(word_list)        
        if words_in_sentence==1:
            sentence_bb_coord = coord_array[start]
            sentence_bb.append(sentence_bb_coord)
            # cv2.polylines(im,[np.array(sentence_bb_coord)],True,(0,0,255),2)

        else:
            coords = coord_array[start:start+words_in_sentence]
            sentence_bb_coord=[[coords[0,0,0], coords[0,0,1]],
                               [coords[-1,1,0], coords[-1,1,1]],
                               [coords[-1,2,0], coords[-1,2,1]],
                               [coords[0,3,0], coords[0,3,1]]]
            sentence_bb.append(sentence_bb_coord)
            # cv2.polylines(im,[np.array(sentence_bb_coord)],True,(0,0,255),2)
        start += words_in_sentence

    # display_image_in_actual_size(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), im.shape)
    return np.array(sentence_bb)

def get_rotated_bb(textdetection_response):
    img_width = textdetection_response.full_text_annotation.pages[0].width
    img_height = textdetection_response.full_text_annotation.pages[0].height
    img_dims = (img_height, img_width)
    

    coord_array, word_list = get_eng_words_bb(textdetection_response)
    # rotation_matrix = get_rotation_matrix(-10, img_dims) #this could be replacement for m=1, (1)
    # coord_array = rotate_bb(coord_array, rotation_matrix) # (2)

    bb_chars, char_list = get_eng_chars_bb(textdetection_response)
    # bb_chars = rotate_bb(bb_chars, rotation_matrix) # (3)

    coord_for_slope = coord_array[:]
    bb_slopes = np.asarray(list(map(get_invoice_orientation,coord_for_slope))) # can vectorize this function

    if np.unique(bb_slopes).size == 1:
        invoice_orientation = np.unique(bb_slopes)
        print("invoice_orientation actual", invoice_orientation)
    else:
        bb_slopes = np.abs(bb_slopes)        
        invoice_orientation = np.mean(bb_slopes)
        print("invoice_orientation actual", invoice_orientation)
    
    rotation_matrix = get_rotation_matrix(-1.0 * invoice_orientation, img_dims)
    coord_for_slope_rotated = rotate_bb(coord_for_slope, rotation_matrix)
    
    # image = cv2.imread(p)
    # plot_eng_words_bb(image, coord_array)
    # rotated = imutils.rotate_bound(image, -invoice_orientation)
    # display_image_in_actual_size(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), rotated.shape)
    
    text_orientation_0_or_180 = np.array(list(map(get_orientation, coord_for_slope_rotated))) # PPPPP6666C or Ɔ9999ԀԀԀԀԀ?
    text_orientation_0_or_180 = np.bincount(text_orientation_0_or_180).argmax() # get mode of orientations
    print("0 or 180 degrees: ", text_orientation_0_or_180)

    if text_orientation_0_or_180==180:
        rotation_matrix = get_rotation_matrix(180-invoice_orientation, img_dims)
        bb_chars = rotate_bb(bb_chars, rotation_matrix)
        # rotated = imutils.rotate_bound(image, 180-invoice_orientation)
        # display_image_in_actual_size(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), rotated.shape)

    else:
        bb_chars = rotate_bb(bb_chars, rotation_matrix)
        
    coord_array = get_sentence_bb(word_list, bb_chars)
    w, _ = preprocess(textdetection_response.text_annotations[0].description)
    coord_array = get_sentence_bb1(w, coord_array)
    return coord_array, word_list, bb_chars, char_list, w, img_dims

def get_invoice_orientation(coord_for_slope):
    epsilon = 0.00001
    dely = coord_for_slope[0][1]-coord_for_slope[1][1]
    delx = (coord_for_slope[0][0]-coord_for_slope[1][0]) if (coord_for_slope[0][0]-coord_for_slope[1][0]) != 0 else epsilon # for divide by 0
    invoice_orientation = np.arctan([dely/delx]) * 180 / np.pi
    return invoice_orientation

def get_rotation_matrix(angle, img_dims):
    # grab the dimensions of the image and then determine the center
    (h, w) = img_dims
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return M

def rotate_bb(bb, rotation_matrix):
    g_sbb = bb
    g_sbb_flat = g_sbb.reshape(-1,2) # transforming all sentence bb (num_sentences x 4 x 2) to 
                                     # (num_sentences x 2) for dot product, 
                                     # so that all bb can be transformed at once

    # unraveling to (num_sentences x 4 x 2)
    bb_transformed = cv2.transform(np.array([g_sbb_flat]), rotation_matrix)[0].reshape(-1,4,2)
    return bb_transformed

def get_orientation(coord):
    """get_invoice_orientation can only tell the orientation of invoice,
    this function gives the direction in which the text and face in the invoice is actually oriented.
    
    EX. PPPPP6666C AND Ɔ9999ԀԀԀԀԀ both have invoice orientation 0 but later text is facing in 180 degrees direction"""
    gutter = 10
    x0 = coord[0][0]
    x1 = coord[1][0]
    y0 = coord[0][1]
    y1 = coord[1][1]
    if (x1<=x0+gutter) & (x1>=x0-gutter):
        if (y0<y1):
            return 90
        else:
            return 270
        
    else:
        if x0<x1:
            return 0
        else:
            return 180

def ocr_using_google_api(image):
    '''
    This function uses Google Vision API for Text Detection
    Args :
        image : Input image 
    Returns:
        pd.DataFrame having coordinates of each text box along with detected
        text inside it.
    '''
    textdetection_response = ocr_processing(image)
    print(textdetection_response.text_annotations[0].description)
    coord_array, word_list, _, _, w, img_dims = get_rotated_bb(textdetection_response)
    coord_object = pd.DataFrame(coord_array[:,(0,2),:].reshape(-1,4), columns=['xmin','ymin','xmax','ymax'])
    coord_object['Object'] = w

    return coord_object, img_dims



if __name__ == '__main__':
    # coord_object = ocr_using_google_api('D:\\Adoor\\hack\\t2\\data\\\HomeCentre\\5.jpg')
    # coord_object.to_csv('ab.csv')
    # print(coord_object)
    lis = [
        'D:\\Adoor\\hack\\t2\\data\\WowMomos\\10.png'#,
        # 'D:\\Adoor\\hack\\t2\\data\\VBsignature\\8.png',
        # 'D:\\Adoor\\hack\\t2\\data\\VBsignature\\9.png',
        # 'D:\\Adoor\\hack\\t2\\data\\VBsignature\\10.png'
    ]
    for i in lis:#glob('D:\\Adoor\\hack\\t2\\data\\*\\*.jpg'):
        print()
        print(i)
        path = os.path.normpath(i)
        base = '/'.join(path.split(os.sep)[:-1])
        csv_path = base + '/' + '_'.join(path.split(os.sep)[-2:]).split('.')[0] + '.csv'
        
        coord_object = ocr_using_google_api(i)
        coord_object.to_csv(csv_path, index=False)