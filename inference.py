import os
import warnings
import base64
import numpy as np
import pandas as pd
import pickle
import cv2
from datefinder import find_dates
from ocr import ocr_using_google_api
from grapher import ObjectTree, Graph
warnings.filterwarnings('ignore')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './capfirst-service-role.json'

class Predictor():
    def __init__(self):
        self.model_bin = None
        self.model_mult = None
        self.object_map = None
        self.tree = None
        self.graph = None
        self.image = None
        self.X = None
        self.img_dims = None
        self.label_mapping = {
            0: 'Other',      
            1: 'Store Name', 
            2: 'Address',               
            3: 'Invoice Number Key',    
            4: 'Invoice Number Value', 
            5: 'Date-Time Key',        
            6: 'Date-Time Value', #
            7: 'Item Key', 
            8: 'Item Value', 
            9: 'Amount Key', 
            10: 'Amount Value' #
        }
        self.entity_classifier = None
        self.company_name_model = None
        self.company_name_cv = None
        self.invoice_model = None
        self.invoice_cv = None
    
    def load_models(self,
                    entity_classifier_mod=r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\models\model_1_2.sav', 
                    company_name_mod=r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\models\company_nb.sav', 
                    company_name_cv=r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\models\company_nb_cv.sav',
                    invoice_no_mod=r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\models\invoice_nb.sav', 
                    invoice_no_cv=r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\models\invoice_nb_cv.sav'):
        '''
        Function to load a trained model in sklearn

        Args:
            company_address_mod: str, filepath to .pkl file for address 
                                        recognition model

            invoice_no_mod: str, filepath to .pkl file for invoice number
                                    recognition model
        '''
        self.entity_classifier = pickle.load(open(entity_classifier_mod, 'rb'))

        # with open(entity_classifier_mod, 'rb') as f:
        #     self.entity_classifier = pickle.load(f)
        self.company_name_model = pickle.load(open(company_name_mod, 'rb'))
        self.company_name_cv = pickle.load(open(company_name_cv, 'rb'))
        self.invoice_model = pickle.load(open(invoice_no_mod, 'rb'))
        self.invoice_cv = pickle.load(open(invoice_no_cv, 'rb'))

        print('Models loaded from disk.')

    def _ocr(self, image):
        '''
        Function to perform OCR and generate Object Map

        Args:
            image: np.array
        
        Returns:
            object map: pd.DataFrame with columns xmin, ymin, xmax, ymax
        '''
        print('Performing OCR...\n')
        self.object_map, self.img_dims = ocr_using_google_api(image)
        self.image = image

    def _generate_graph_embeddings(self):
        '''
        This function first generates an object tree using the ObjectTree Class
        and then generates a graph using the Graph Class. 

        The output of this function is a feature array along with label array
        (if any).
        '''
        self.tree = ObjectTree()
        self.graph = Graph()

        # Generate a graph dictionary 
        self.tree.read(self.object_map, self.image)
        graph_dict, text_list, coords_arr = self.tree.connect(plot=False, export_df=True)

        # Generate the Adjacency Matrix (A) and the Feature Matrix (X)
        A, X = self.graph.make_graph_data(graph_dict=graph_dict, 
                                    text_list=text_list, 
                                    coords_arr=coords_arr, 
                                    img_dims=self.img_dims)
        
        # transform the feature matrix by muptiplying with the Adjacency Matrix
        X_transformed = np.matmul(A, X)
        
        # form new feature matrix
        self.X = np.concatenate((X, X_transformed), axis=1)

    @staticmethod
    def get_dates(string):
        '''
        Wrapper for datefinder.find_dates function

        Returns:
            date_time string found in input string
        '''
        for match in find_dates(string):
            date_time = match
        
        return date_time

    def perform_elimination(self, label_wise_objects):
        # ================== for Invoice Number Value ==========================
        # invoice_no_list = []
        # if len(label_wise_objects[4]) != 0:
        #     for possible_invoice_no in label_wise_objects[4]:
        #         invoice_cv_text = self.invoice_cv.transform([possible_invoice_no])
        #         if self.invoice_model.predict(invoice_cv_text) == 1:
        #             invoice_no_list.append(possible_invoice_no)
        
        # else:
        #     pass
        # label_wise_objects[4] = invoice_no_list
        # ----------------------------------------------------------------------

        
        # =================== for Date Time ====================================
        date_time_list = []
        if len(label_wise_objects[6]) != 0:
            for possible_date_time in label_wise_objects[6]:
                if len(list(find_dates(possible_date_time))) != 0:
                    date_time_list.append(list(find_dates(possible_date_time))[0])
        
        else:
            text_corpus = self.object_map.Object.values
            for possible_date_time in text_corpus:
                if len(list(find_dates(possible_date_time))) !=0:
                    date_time_list.append(list(find_dates(possible_date_time))[0])
        
        label_wise_objects[6] = date_time_list
        # ----------------------------------------------------------------------

        # ============= for amount value =======================================
        try:
            amount_value_list = []
            for possible_amount in label_wise_objects[10]:
                if possible_amount.replace('.', '', 1).isnumeric():
                    amount_value_list.append(possible_amount)
        except:
            label_wise_objects[10] = amount_value_list
        # ======================================================================

        # ============== for company name ======================================
        company_name_list = []
        for possible_company_name in label_wise_objects[1]:
            company_cv_text = self.company_name_cv.transform([possible_company_name])
            if self.company_name_model.predict(company_cv_text) == 1:
                company_name_list.append(possible_company_name)
        
        label_wise_objects[1] = company_name_list

        # ----------------------------------------------------------------------

        return label_wise_objects

    def infer(self, image):
        '''
        This function implements the entire pipeline for information extraction
        from the input document image.

        Prerequisites: All the ML Models must be loaded 

        Args:
            Image, cv2

        Returns:
            JSON Object
        '''

        # perform OCR
        # print(image)
        # retval, buffer = cv2.imencode('.jpg', image)
        # jpg_as_text = base64.b64encode(buffer)
        # print(jpg_as_text)
        
        self._ocr(image)

        # Generate Graph Embeddings
        self._generate_graph_embeddings()

        # check if all models are available
        # assert None not in {self.entity_classifier, 
        #                     self.address_model, 
        #                     self.address_cv, 
        #                     self.invoice_model, 
        #                     self.invoice_cv}, "One or more of the required \
        #                         models has not been loaded properly. Please see\
        #                         `load_models() function which is a pre-requisite."
        
        preds = self.entity_classifier.predict(self.X)
        self.object_map['label'] = preds

        label_wise_objects = dict(self.object_map.groupby('label')['Object'].apply(list))
        retained_objects = self.perform_elimination(label_wise_objects)

        try:
            retained_objects.pop(0)
        except:
            pass
        
        try:
            retained_objects.pop(3)
        except:
            pass

        try:
            retained_objects.pop(5)
        except:
            pass

        try:
            retained_objects.pop(7)
        except:
            pass
        
        try:
            retained_objects.pop(9)
        except:
            pass

        output_dict = {}
        for key, value in retained_objects.items():
            output_dict[self.label_mapping[key]] = str(value)

        return output_dict


if __name__ == '__main__':
    def load_image(img_filename):
        with open(img_filename, 'rb') as image_file:
            content = bytearray(image_file.read())
            image_file.close()
        return content

    predictor = Predictor()
    # image = cv2.imread(r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\final_data\ColdStoneCreamery\7.png', 0)
    # df = pd.read_csv(r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\final_data\ColdStoneCreamery\7.csv')
    
    predictor.load_models()
    predictor.infer(load_image(r'C:\Users\Think Analytics\Desktop\Side_Projects\Graph-Convolution-on-Structured-Documents-master\final_data\ColdStoneCreamery\7.png'))