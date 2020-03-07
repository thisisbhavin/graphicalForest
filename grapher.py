import numpy as np
import pandas as pd
import cv2
import os
import sys


# for making adjacency matrix
import networkx as nx

np.set_printoptions(threshold=sys.maxsize)
class ObjectTree:	
	'''
		Description:
		-----------
			This class is used to generate a dictionary of lists that contain
			the graph structure:
				{src_id: [dest_id1, dest_id2, ..]}
			and return the list of text entities in the input document
		
		Example use:
		-----------
			>> connector = ObjectTree(label_column='label')
			>> connector.read(object_map_df, img)
			>> df, obj_list = connector.connect(plot=False, export_df=False)
	'''

	def __init__(self, label_column='label'):
		self.label_column = label_column
		self.df = None
		self.img = None
		self.count = 0
	
	def read(self, object_map, image):
		
		'''
			Function to ensure the data is in correct format and saves the 
			dataframe and image as class properties

			Args:
				object_map: pd.DataFrame, having coordinates of bounding boxes,
										  text object and label
				
				image: np.array, black and white cv2 image

			Returns:
				None

		'''

		assert type(object_map) == pd.DataFrame,f'object_map should be of type \
			{pd.DataFrame}. Received {type(object_map)}'
		assert type(image) == np.ndarray,f'image should be of type {np.ndarray} \
			. Received {type(image)}'

		assert 'xmin' in object_map.columns, '"xmin" not in object map'
		assert 'xmax' in object_map.columns, '"xmax" not in object map'
		assert 'ymin' in object_map.columns, '"ymin" not in object map'
		assert 'ymax' in object_map.columns, '"ymax" not in object map'
		assert 'Object' in object_map.columns, '"Object" column not in object map'
		assert self.label_column in object_map.columns, \
						f'"{self.label_column}" does not exist in the object map'

		# check if image is greyscale
		assert image.ndim == 2, 'Check if the read image is greyscale.'

		# drop unneeded columns
		required_cols = {'xmin', 'xmax', 'ymin', 'ymax', 'Object', 
							self.label_column}
		un_required_cols = set(object_map.columns) - required_cols
		object_map.drop(columns=un_required_cols, inplace=True)
		
		self.df = object_map
		self.img = image
		return 
			 
	def connect(self, plot=False, export_df=False):
		
		'''
			This method implements the logic to generate a graph based on
			visibility. If a horizontal/vertical line can be drawn from one
			node to another, the two nodes are connected.

			Args:
				plot (default=False):
					bool, whether to plot the graph;
					the graph is plotted in at path ./grapher_outputs/plots

				export_df (default=False):
					bool, whether to export the dataframe containing graph 
					information;
					the dataframe is exported as csv to path 
					./grapher_outputs/connections
		'''
		df, img = self.df, self.img

		# check if object map was successfully read by .read() method
		try:
			if len(df) == 0:
				return
		except:
			return
		
		# initialize empty df to store plotting coordinates
		df_plot = pd.DataFrame()
		
		# initialize empty lists to store coordinates and distances
		# ================== vertical======================================== #
		distances, nearest_dest_ids_vert = [], []
		
		x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, \
		y_dest_coords_vert = [], [], [], []

		# ======================= horizontal ================================ #
		lengths, nearest_dest_ids_hori = [], []
		
		x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, \
		y_dest_coords_hori = [], [], [], []

		for src_idx, src_row in df.iterrows():
			
			# ================= vertical ======================= #
			src_range_x = (src_row['xmin'], src_row['xmax'])
			src_center_y = (src_row['ymin'] + src_row['ymax'])/2

			dest_attr_vert = []

			# ================= horizontal ===================== #
			src_range_y = (src_row['ymin'], src_row['ymax'])
			src_center_x = (src_row['xmin'] + src_row['xmax'])/2
			
			dest_attr_hori = []

			################ iterate over destination objects #################
			for dest_idx, dest_row in df.iterrows():
				# flag to signal whether the destination object is below source
				
				is_beneath = False
				if not src_idx == dest_idx:
					# ==================== vertical ==========================#
					dest_range_x = (dest_row['xmin'], dest_row['xmax'])
					dest_center_y = (dest_row['ymin'] + dest_row['ymax'])/2
					
					height = dest_center_y - src_center_y

					# consider only the cases where destination object lies 
					# below source
					if dest_center_y > src_center_y:
						# check if horizontal range of dest lies within range 
						# of source

						# case 1
						if dest_range_x[0] <= src_range_x[0] and \
							dest_range_x[1] >= src_range_x[1]:
							
							x_common = (src_range_x[0] + src_range_x[1])/2
							
							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)
							
							is_beneath = True

						# case 2
						elif dest_range_x[0] >= src_range_x[0] and \
							dest_range_x[1] <= src_range_x[1]:
							
							x_common = (dest_range_x[0] + dest_range_x[1])/2
							
							line_src = (x_common, src_center_y)
							line_dest = (x_common, dest_center_y)
							
							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)
							
							is_beneath = True

						# case 3
						elif dest_range_x[0] <= src_range_x[0] and \
							dest_range_x[1] >= src_range_x[0] and \
								dest_range_x[1] < src_range_x[1]:

							x_common = (src_range_x[0] + dest_range_x[1])/2

							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)

							is_beneath = True

						# case 4
						elif dest_range_x[0] <= src_range_x[1] and \
							dest_range_x[1] >= src_range_x[1] and \
								dest_range_x[0] > src_range_x[0]:
							
							x_common = (dest_range_x[0] + src_range_x[1])/2
							
							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)

							is_beneath = True
			
				if not is_beneath:
					# ======================= horizontal ==================== #
						dest_range_y = (dest_row['ymin'], dest_row['ymax'])
						# get center of destination NOTE: not used
						dest_center_x = (dest_row['xmin'] + dest_row['xmax'])/2
						
						# get length from destination center to source center
						if dest_center_x > src_center_x:
							length = dest_center_x - src_center_x
						else:
							length = 0
						
						# consider only the cases where the destination object 
						# lies to the right of source
						if dest_center_x > src_center_x:
							#check if vertical range of dest lies within range 
							# of source
							
							# case 1
							if dest_range_y[0] >= src_range_y[0] and \
								dest_range_y[1] <= src_range_y[1]:
								
								y_common = (dest_range_y[0] + dest_range_y[1])/2

								line_src = (src_center_x, y_common)
								line_dest = (dest_center_x, y_common)
								
								attributes = (dest_idx, line_src, line_dest, length)
								dest_attr_hori.append(attributes)

							# case 2
							if dest_range_y[0] <= src_range_y[0] and \
								dest_range_y[1] <= src_range_y[1] and \
									dest_range_y[1] > src_range_y[0]:
								
								y_common = (src_range_y[0] + dest_range_y[1])/2

								line_src = (src_center_x, y_common)
								line_dest = (dest_center_x, y_common)

								attributes = (dest_idx, line_src, line_dest, length)
								dest_attr_hori.append(attributes)

							# case 3
							if dest_range_y[0] >= src_range_y[0] and \
								dest_range_y[1] >= src_range_y[1] and \
									dest_range_y[0] < src_range_y[1]:

								y_common = (dest_range_y[0] + src_range_y[1])/2

								line_src = (src_center_x, y_common)
								line_dest = (dest_center_x, y_common)

								attributes = (dest_idx, line_src, line_dest, length)
								dest_attr_hori.append(attributes)

							# case 4
							if dest_range_y[0] <= src_range_y[0] \
								and dest_range_y[1] >= src_range_y[1]:

								y_common = (src_range_y[0] + src_range_y[1])/2

								line_src = (src_center_x, y_common)
								line_dest = (dest_center_x, y_common)

								attributes = (dest_idx, line_src, line_dest, length)
								dest_attr_hori.append(attributes)

			# sort list of destination attributes by height/length at position 
			# 3 in tuple
			dest_attr_vert_sorted = sorted(dest_attr_vert, key = lambda x: x[3])
			dest_attr_hori_sorted = sorted(dest_attr_hori, key = lambda x: x[3])
			
			# append the index and source and destination coords to draw line 
			# ==================== vertical ================================= #
			if len(dest_attr_vert_sorted) == 0:
				nearest_dest_ids_vert.append(-1)
				x_src_coords_vert.append(-1)
				y_src_coords_vert.append(-1)
				x_dest_coords_vert.append(-1)
				y_dest_coords_vert.append(-1)
				distances.append(0)
			else:
				nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
				x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
				y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
				x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
				y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
				distances.append(dest_attr_vert_sorted[0][3])

			# ========================== horizontal ========================= #
			if len(dest_attr_hori_sorted) == 0:
				nearest_dest_ids_hori.append(-1)
				x_src_coords_hori.append(-1)
				y_src_coords_hori.append(-1)
				x_dest_coords_hori.append(-1)
				y_dest_coords_hori.append(-1)
				lengths.append(0)			
			
			else:
			# try and except for the cases where there are vertical connections
			# still to be made but all horizontal connections are accounted for
				try:
					nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
				except:
					nearest_dest_ids_hori.append(-1)
				
				try:
					x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
				except:
					x_src_coords_hori.append(-1)

				try:
					y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
				except:
					y_src_coords_hori.append(-1)

				try:
					x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
				except:
					x_dest_coords_hori.append(-1)

				try:
					y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
				except:
					y_dest_coords_hori.append(-1)

				try:
					lengths.append(dest_attr_hori_sorted[0][3])
				except:
					lengths.append(0)			

		# ==================== vertical ===================================== #
		# create df for plotting lines
		df['below_object'] = df.loc[nearest_dest_ids_vert, 'Object'].values  

		# add distances column
		df['below_dist'] = distances
		
		# add column containing index of destination object
		df['below_obj_index'] = nearest_dest_ids_vert

		# add coordinates for plotting
		df_plot['x_src_vert'] = x_src_coords_vert
		df_plot['y_src_vert'] = y_src_coords_vert
		df_plot['x_dest_vert'] = x_dest_coords_vert
		df_plot['y_dest_vert'] = y_dest_coords_vert
		
		# df.fillna('NULL', inplace = True)

		# ==================== horizontal =================================== #
		# create df for plotting lines
		df['side_object'] = df.loc[nearest_dest_ids_hori, 'Object'].values

		# add lengths column
		df['side_length'] = lengths

		# add column containing index of destination object
		df['side_obj_index'] = nearest_dest_ids_hori

		# add coordinates for plotting
		df_plot['x_src_hori'] = x_src_coords_hori
		df_plot['y_src_hori'] = y_src_coords_hori
		df_plot['x_dest_hori'] = x_dest_coords_hori
		df_plot['y_dest_hori'] = y_dest_coords_hori

		########################## concat df and df_plot ######################
		
		df_merged = pd.concat([df, df_plot], axis=1)
		
		# if an object has more than one parent above it, only the connection 
		# with the smallest distance is retained and the other distances are 
		# replaced by '-1' to get such objects, group by 'below_object' column 
		# and use minimum of 'below_dist'
		
		# ======================= vertical ================================== #
		groups_vert = df_merged.groupby('below_obj_index')['below_dist'].min()
		# groups.index gives a list of the below_object text and groups.values 
		# gives the corresponding minimum distance
		groups_dict_vert = dict(zip(groups_vert.index, groups_vert.values))

		# ======================= horizontal ================================ #
		groups_hori = df_merged.groupby('side_obj_index')['side_length'].min()
		# groups.index gives a list of the below_object text and groups.values 
		# gives the corresponding minimum distance
		groups_dict_hori = dict(zip(groups_hori.index, groups_hori.values))
		
			
		revised_distances_vert = []
		revised_distances_hori = []

		rev_x_src_vert, rev_y_src_vert, rev_x_dest_vert, rev_y_dest_vert = \
																[], [], [], []
		rev_x_src_hori, rev_y_src_hori, rev_x_dest_hori, rev_y_dest_hori = \
																[], [], [], []
		
		for idx, row in df_merged.iterrows():
			below_idx = row['below_obj_index']
			side_idx = row['side_obj_index']

			# ======================== vertical ============================= #
			if row['below_dist'] > groups_dict_vert[below_idx]:
				revised_distances_vert.append(-1)
				rev_x_src_vert.append(-1)
				rev_y_src_vert.append(-1)
				rev_x_dest_vert.append(-1)
				rev_y_dest_vert.append(-1)

			else:
				revised_distances_vert.append(row['below_dist'])
				rev_x_src_vert.append(row['x_src_vert'])
				rev_y_src_vert.append(row['y_src_vert'])
				rev_x_dest_vert.append(row['x_dest_vert'])
				rev_y_dest_vert.append(row['y_dest_vert'])

			# ========================== horizontal ========================= #
			if row['side_length'] > groups_dict_hori[side_idx]:
				revised_distances_hori.append(-1)
				rev_x_src_hori.append(-1)
				rev_y_src_hori.append(-1)
				rev_x_dest_hori.append(-1)
				rev_y_dest_hori.append(-1)

			else:
				revised_distances_hori.append(row['side_length'])
				rev_x_src_hori.append(row['x_src_hori'])
				rev_y_src_hori.append(row['y_src_hori'])
				rev_x_dest_hori.append(row['x_dest_hori'])
				rev_y_dest_hori.append(row['y_dest_hori'])

		# store in dataframe
		# ============================ vertical ============================= #
		df['revised_distances_vert'] = revised_distances_vert
		df_merged['x_src_vert'] = rev_x_src_vert
		df_merged['y_src_vert'] = rev_y_src_vert
		df_merged['x_dest_vert'] = rev_x_dest_vert
		df_merged['y_dest_vert'] = rev_y_dest_vert

		# ======================== horizontal =============================== #
		df['revised_distances_hori'] = revised_distances_hori
		df_merged['x_src_hori'] = rev_x_src_hori
		df_merged['y_src_hori'] = rev_y_src_hori
		df_merged['x_dest_hori'] = rev_x_dest_hori
		df_merged['y_dest_hori'] = rev_y_dest_hori
		
		
		# plot image if plot==True 
		if plot == True:
		
			# make folder to store output
			if not os.path.exists('grapher_outputs'):
				os.makedirs('grapher_outputs')	
			
			# subdirectory to store plots
			if not os.path.exists('./grapher_outputs/plots'):
				os.makedirs('./grapher_outputs/plots')
			
			# check if image exists in folder
			try: 
				if len(img) == None:
					pass
			except: 
				pass
			
			# plot if image exists
			else:
				for idx, row in df_merged.iterrows():
					cv2.line(img, 
							(int(row['x_src_vert']), int(row['y_src_vert'])), 
							(int(row['x_dest_vert']), int(row['y_dest_vert'])), 
							(0,0,255), 2)
					
					cv2.line(img, 
							(int(row['x_src_hori']), int(row['y_src_hori'])), 
							(int(row['x_dest_hori']), int(row['y_dest_hori'])), 
							(0,0,255), 2)

				# write image in same folder
				PLOT_PATH = \
					'./grapher_outputs/plots/' + 'object_tree_' + str(self.count) + '.jpg'
				cv2.imwrite(PLOT_PATH, img)


		# export dataframe with destination objects to csv in same folder
		if export_df == True:
			
			# make folder to store output
			if not os.path.exists('grapher_outputs'):
				os.makedirs('grapher_outputs')			
			
			# subdirectory to store plots
			if not os.path.exists('./grapher_outputs/connections'):
				os.makedirs('./grapher_outputs/connections')

			CSV_PATH = \
				'./grapher_outputs/connections/' + 'connections_' + str(self.count) + '.csv'
			df.to_csv(CSV_PATH, index = None)
			 
		# convert dataframe to dict:
		# {src_id: dest_1, dest_2, ..}

		graph_dict = {}
		for src_id, row in df.iterrows():
			if row['below_obj_index'] != -1:
				if src_id in graph_dict.keys():
					graph_dict[src_id].append(row['below_obj_index'])
				else:
					graph_dict[src_id] = [row['below_obj_index']]

			if row['side_obj_index'] != -1:
				if src_id in graph_dict.keys():
					graph_dict[src_id].append(row['side_obj_index'])
				else:
					graph_dict[src_id] = [row['side_obj_index']]
			
		
		return graph_dict, df['Object'].tolist(), df[['xmin',
														'ymin', 
														'xmax', 
														'ymax']].values


class Graph:
	'''
		This class generates a padded adjacency matrix and a feature matrix
	'''
	def __init__(self, max_nodes=50):
		self.max_nodes = max_nodes
		self.image = None
		return

	# def make_graph(self, graph_dict):
	# 	'''
	# 		Function to make networkx graph

	# 		Args:
	# 			graph_dict: dict of lists, 
	# 						{src_id: [dest_id]}

				
	# 		Returns:
	# 			G: 
	# 				Padded adjacency matrix of size (max_nodes, max_nodes)

	# 			feats:
	# 				Padded feature matrix of size (max_nodes, m)
	# 				(m: dimension of node text vector)
	# 	'''
	# 	G = nx.from_dict_of_lists(graph_dict)

	# 	return G

	def _get_text_features(self, data):

		'''
			Args:
				str, input data
				
			Returns: 
				np.array, shape=(35,);
				an array of the text converted to features
				
		'''
		assert type(data) == str, f'Expected type {str}. Received {type(data)}.'

		data = r'{}'.format(data)

		n_lower = 0
		n_upper = 0
		n_digits = 0

		# make a mapping dict of special characters
		mapping_dict = {
			'-': 0, 
			'.': 1, 
			',': 2, 
			'/': 3, 
			'\\': 4,
			':': 5
		}
		initial_len_mapping_dict = len(mapping_dict)
		# add the alphabet as the keys to the mapping dict
		for idx, char in enumerate('abcedfghijklmnopqrstuvwxyz'):
			mapping_dict[char] = idx + initial_len_mapping_dict

		# get number of lower and upper case letters
		for char in data:
			if char.islower():
				n_lower += 1
			
			if char.isupper():
				n_upper += 1

			if char.isdigit():
				n_digits += 1

		# concat to form the vector in form:
		# | 0-29: character mapping | n_lower | n_upper | n_digits |

		vector_arr = np.zeros(35)
		for char in data.lower():
			if char in mapping_dict.keys():
				vector_arr[mapping_dict[char]] += 1
			else:
				pass

		vector_arr[32] = n_lower
		vector_arr[33] = n_upper
		vector_arr[34] = n_digits

		return vector_arr

	def _get_positional_embeddings(self, coordinate_arr):
		'''
		This method returns the normalized array of coordinates as per the shape
		of the image

		Args:
			coordinate_arr: [`xmin`, `ymin`, `xmax`, `ymax`]

		Returns:
			Normalized coordinate array
		'''
		image_height, image_width = self.image.shape

		normalized_coordinate_list = []

		normalized_coordinate_list.append(coordinate_arr[0]/image_width)
		normalized_coordinate_list.append(coordinate_arr[1]/image_height)
		normalized_coordinate_list.append(coordinate_arr[2]/image_width)
		normalized_coordinate_list.append(coordinate_arr[3]/image_height)

		return np.array(normalized_coordinate_list)

	def _pad_adj(self, adj):
		'''
			This method resizes the input Adjacency matrix to shape 
			(self.max_nodes, self.max_nodes)

			adj: 
				2d numpy array
				adjacency matrix
		'''
		
		assert adj.shape[0] == adj.shape[1], f'The input adjacency matrix is \
			not square and has shape {adj.shape}'
		
		# get n of nxn matrix
		n = adj.shape[0]
		
		if n < self.max_nodes:
			target = np.zeros(shape=(self.max_nodes, self.max_nodes))

			# fill in the target matrix with the adjacency
			target[:adj.shape[0], :adj.shape[1]] = adj
			
		elif n > self.max_nodes:
			# cut away the excess rows and columns of adj
			target = adj[:self.max_nodes, :self.max_nodes]
			
		else:
			# do nothing
			target = adj
			
		return target
	
	def _pad_text_features(self, feat_arr):
		'''
			This method pads the feature matrix to size 
			(self.max_nodes, feat_arr.shape[1])
		'''
		target = np.zeros(shape=(self.max_nodes, feat_arr.shape[1]))

		if self.max_nodes > feat_arr.shape[0]:
			target[:feat_arr.shape[0], :feat_arr.shape[1]] = feat_arr

		elif self.max_nodes < feat_arr.shape[0]:
			target = feat_arr[:self.max_nodes, feat_arr.shape[1]]

		else: 
			target = feat_arr

		return target


	def make_graph_data(self, graph_dict, text_list, coords_arr, img):
		'''
			Function to make an adjacency matrix from a networkx graph object
			as well as padded feature matrix

			Args:
				G: networkx graph object
				
				text_list: list,
							of text entities:
							['Tax Invoice', '1/2/2019', ...]
				
				coords_arr: np.array, of coordinates for each node

				img: cv2 image of the document

			Returns:
				A: Adjacency matrix as np.array

				X: Feature matrix as numpy array for input graph
		'''
		self.image = img
		
		G = nx.from_dict_of_lists(graph_dict)
		adj_sparse = nx.adjacency_matrix(G)

		# preprocess the sparse adjacency matrix returned by networkx function
		A = np.array(adj_sparse.todense())
		
		# NOTE: this was removed so as to not pad the adjacency matrix
		# A = self._pad_adj(A)

		# preprocess the list of text entities
		feat_list = list(map(self._get_text_features, text_list))
		feat_arr = np.array(feat_list)
		
		# get positional embeddings
		pos_list = list(map(self._get_positional_embeddings, coords_arr))
		pos_arr = np.array(pos_list)

		# concatenate positional features with text features
		feat_arr = np.concatenate((pos_arr, feat_arr), axis=1)

		# NOTE: this was removed so as to not pad the feature matrix
		X = feat_arr
		# X = self._pad_text_features(X)

		return A, X

	



if __name__ == "__main__":
	print(os.getcwd())
	df = pd.read_csv(r'C:\Users\Think Analytics\Desktop\Side_Projects\graph_test\object_map.csv')
	img = cv2.imread(r'C:\Users\Think Analytics\Desktop\Side_Projects\graph_test\deskew.jpg', 0)
	
	tree = ObjectTree()
	tree.read(df, img)
	
	graph_dict, text_list, coords_arr = tree.connect(plot=True, export_df=True)
	
	print(graph_dict)
	print('\n--------------------------------------------------------------\n')

	graph = Graph()
	A, X = graph.make_graph_data(graph_dict, text_list, coords_arr, img)
	

	print(A)
	print('-----------------------------------------------------------------\n')
	print(X)

	np.save('./A.npy', A)
	np.save('./X.npy', X)

