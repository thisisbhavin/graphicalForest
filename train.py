import warnings
import pandas as pd
import numpy as np
import cv2
from glob import glob
from grapher import ObjectTree, Graph
warnings.filterwarnings('ignore')

def get_train_data(data_folder_path):
    # ============= get paths for images and object maps ===========================
    image_jpg_path_glob = data_folder_path + r'\*\*.jpg'
    image_png_path_glob = data_folder_path + r'\*\*.png'
    csv_path_glob = data_folder_path + r'\*\*.csv'
    
    list_img_paths = glob(image_jpg_path_glob) + \
        glob(image_png_path_glob)
    list_csv_paths = glob(csv_path_glob)
    # ------------------------------------------------------------------------------

    # =============== initialize the ObjectTree and Graph Objects ==================
    tree = ObjectTree()
    graph = Graph()
    # ------------------------------------------------------------------------------

    # === generate graph embeddings for each document ============================== 
    data_df = pd.DataFrame(columns=['features', 'label'])
    count, skip_count = 0, 0


    for image_path, csv_path in zip(list_img_paths, list_csv_paths):

        img = cv2.imread(image_path, 0)
        object_map = pd.read_csv(csv_path)

        # drop rows with nans 
        object_map.dropna(inplace=True, how='any')

        try:
            # generate object tree
            tree.read(object_map, img)
            graph_dict, text_list, coords_arr = tree.connect(plot=True, export_df=True)

            # make graph data
            A, X = graph.make_graph_data(graph_dict, text_list, coords_arr, img)

            # transform the feature matrix by muptiplying with the Adjacency Matrix
            X_transformed = np.matmul(A, X)
            
            # form new feature matrix
            X = np.concatenate((X, X_transformed), axis=1)

            # get the labels
            y = object_map.label.values

            if count == 0:
                X_train = X
                y_train = y

            else:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

            print('Finished processing image {} of {}...'.format(count, len(list_img_paths)))

        except Exception:
            skip_count += 1
            print('Skipping Graph Generation...')
            
        count += 1
        
        

    print('Finished generating Graph Dataset for {} documents. Skipped {}.'\
        .format(count, skip_count))
    
    return X_train, y_train



if __name__ == '__main__':
    X_train, y_train = get_train_data(r'./final_data')
    
    print(X_train.shape, y_train.shape)