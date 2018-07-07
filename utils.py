import os
import numpy as np
import random
import config as config
import json	
import cv2
import matplotlib.pyplot as plt
import math
import tensorflow as tf

class readData:

	def __init__(self, dir):

		self.dir = dir
		self.trainPath = os.path.join(dir, 'train')
		self.testPath = os.path.join(dir, 'test')
		self.annoPath = os.path.join(dir, 'annotations.json')
		self.otherPath = os.path.join(dir, 'other')
		self.image_size = config.image_size
		self.image_shape_0 = config.image_height
		self.image_shape_1 = config.image_width
		self.h_ratio = 1.0 * self.image_size / self.image_shape_0
		self.w_ratio = 1.0 * self.image_size / self.image_shape_1
		self.load_data_metrics()

	def load_data_metrics(self):

		#Reading annotation file
		with open(self.annoPath, 'r') as f:
			self.file = json.load(f)

		self.classes = self.file['types']
		self.no_classes = config.no_classes
		self.no_train_images = 0
		self.no_test_images = 0
		self.no_other_images = 0
		self.trainId = []
		self.testId = []
		self.otherId = []

		for imgId in self.file['imgs'].keys():
			path = self.file['imgs'][imgId]['path']
			if 'train' in path:
				self.no_train_images+=1
				self.trainId.append(path.split('.')[0].split('/')[1])
			elif 'test' in path:
				self.no_test_images+=1
				self.testId.append(path.split('.')[0].split('/')[1])
			elif 'other' in path:
				self.no_other_images+=1
				self.otherId.append(path.split('.')[0].split('/')[1])

		self.classes_no = [i for i in range(len(self.classes))]
		self.classes_dict = dict(zip(self.classes, self.classes_no))

	def read_img(self, image_path):

		image = cv2.imread(image_path)
		image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32)
		return image


	def createLabels(self, phase = 'train'):

		if(phase == 'train'):

			imageId = self.trainId
			path_root = self.trainPath

		elif(phase == 'test'):

			imageId = self.testId
			path_root = self.testPath

		label_dict = []
		for imgId in imageId:
			label, no_obj = self.createImageLabel(imgId)

			if(no_obj == 0):
				continue
			
			Id = self.file['imgs'][imgId]['id']
			path = os.path.join(path_root, str(Id)+".jpg")
			label_dict.append({"image_path": path, "id": Id, "label": label})

		return label_dict


	def normXYToGrid(self, x_center, y_center):

		gridCellWidth = self.image_size / config.no_grid
		gridCellHeight = self.image_size / config.no_grid

		gridCellRow = math.floor( y_center / gridCellHeight )
		gridCellCol = math.floor( x_center / gridCellWidth )


		normalizedXCent=(x_center-(gridCellCol * gridCellWidth))/gridCellWidth
		normalizedYCent=(y_center-(gridCellRow * gridCellHeight))/gridCellHeight
		return normalizedXCent, normalizedYCent, gridCellRow, gridCellCol

	def createImageLabel(self, imgId):

		objs = self.file['imgs'][imgId]['objects']
		label = np.zeros((config.no_grid, config.no_grid, 5 + self.no_classes))
		for obj in objs:

			xmin = max(min((float(obj['bbox']['xmin']) - 1) * self.w_ratio, self.image_size - 1), 0)
			xmax = max(min((float(obj['bbox']['xmax']) - 1) * self.w_ratio, self.image_size - 1), 0)
			ymin = max(min((float(obj['bbox']['ymin']) - 1) * self.h_ratio, self.image_size - 1), 0)
			ymax = max(min((float(obj['bbox']['ymax']) - 1) * self.h_ratio, self.image_size - 1), 0)

			w = xmax - xmin
			h = ymax - ymin
			x_center = xmin + w/2.0
			y_center = ymin + h/2.0

			boxes = [x_center, y_center, w, h]

			row = int((y_center / config.image_size) * config.no_grid)
			col = int((x_center / config.image_size) * config.no_grid)

			class_cat = obj['category']
			class_cat_id = self.classes_dict[class_cat]

			if(label[row, col, 0] == 1):
				continue

			label[row, col,0] = 1
			label[row, col,1:5] = boxes
			label[row, col,5+class_cat_id] = 1

		return label, len(objs)

	def yolo_filter_boxes(self, box_confidence, boxes, class_confidence):

		"""
		Input:
		------
		box_confidence : probability of object belonging to this bounding box (no_boxes_per_cell,)
		boxes : predicted boxes (4 * no_boxes_per_cell,)
		class_confidence : confidence (confidence probability of each class) (no_classes,)
		
		Returns:
		--------
		boxes: [x,y, w, h] for the best box
		best_class = index of the best class
		score : (Prob of object belonging to this bounding box * Probability of class ) for best overall box
		"""
		b_c = box_confidence.copy()
		c_c = class_confidence.copy()
		b_c = b_c.reshape(-1,1)
		c_c = c_c.reshape(1,-1)
		mul = b_c * c_c
		best_box_class = np.unravel_index(mul.argmax(), mul.shape)
		best_box, best_class = best_box_class[0], best_box_class[1]


		return boxes[best_box * 4: (best_box + 1) * 4], best_class, mul[best_box, best_class]

	def iou(self, box1, box2):

	    """This function calculates intersection over union (IoU) between box1 and box2
	    
	    Inputs:
	    -------
	    box1 : first box, list object with coordinates (x1, y1, x2, y2)
	    box2 : second box, list object with coordinates (x1, y1, x2, y2)
	    
		Output:
		-------
		iou
	    """

	    #Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
	    xi1 = max(box1[0], box2[0])
	    yi1 = max(box1[1], box2[1])
	    xi2 = min(box1[2], box2[2])
	    yi2 = min(box1[3], box2[3])
	    inter_area = (xi2 - xi1) * (yi2 - yi1)

	    #Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	    box1_area = (box1[3] - box1[1]) * (box1[2]- box1[0])
	    box2_area = (box2[3] - box2[1]) * (box2[2]- box2[0])
	    union_area = (box1_area + box2_area) - inter_area
	    
	    #compute the IoU
	    iou = inter_area / union_area
	    
	    return iou

	def non_maximal_supression(self, boxes, scores, max_boxes = 10, iou_threshold = 0.5):

		"""
		This function suppresses all the boxes with iou lower than a threshold with respect to box with maximum score
		
		Inputs:
		-------
		boxes: list of [xmin, ymin, xmax, ymax] for all the grid cells in the image
		scores: list of best scores
		iou_threshold: threshold of iou for removing the boxes below it
		Returns:
		--------
		rem_box_indexes : indexes of boxes remaining after non maximal supression
		
		"""
		box = tf.convert_to_tensor(np.array(boxes).reshape(-1,4), dtype = tf.float32)
		score = tf.convert_to_tensor(np.array(scores), dtype = tf.float32)
		max_boxes_tensor = tf.constant(max_boxes)
		rem_box_indexes = tf.image.non_max_suppression(box, score, max_boxes_tensor, iou_threshold=iou_threshold)
		return rem_box_indexes.eval()

	def display_bounding_box_batch(self, prediction, image_batch, batch_id):

		fig, ax = plt.subplots()

		for batch in range(config.batch_size):

			predict_ = prediction[batch, :,:,:] 
			image_ = image_batch[batch,:,:,:]
			image_id = batch_id[batch]

			# image_ = cv2.resize(image_batch[batch,:,:,:], (config.image_width, config.image_height))

			class_image = np.zeros(image_.shape)
			all_best_boxes = []
			best_scores = []
			ground_truth_boxes = []
			ground_truth_label,_ = self.createImageLabel(str(image_id))

			for grid_row in range(config.no_grid):
				for grid_col in range(config.no_grid):

					#Appending ground truth boxes if object is there at that grid cell
					if(ground_truth_label[grid_row, grid_col, 0] == 1):
						 ground_truth_boxes.append(ground_truth_label[grid_row, grid_col,1:5])

					boxes_with_prob = predict_[grid_row, grid_col, :]
					box_confidence = boxes_with_prob[:config.no_boxes_per_cell]
					boxes = boxes_with_prob[config.no_boxes_per_cell: 5 * config.no_boxes_per_cell]
					class_confidence = boxes_with_prob[5 * config.no_boxes_per_cell :]


					#Find best score for best class for best bounding box using matrix of shape (no_of_bounding_box_per_cell * no_of_classes) 
					#with each element being (Probability of object * Probability of that class)
					best_box, best_class, best_score = self.yolo_filter_boxes(box_confidence, boxes, class_confidence)

					#Unnormalizing the center, width and height (converting with respect to image upper left and right)
					n_x_cent, n_y_cent, n_w, n_h = best_box

					w = np.square(n_w) * config.image_size / config.no_grid  #square (n_w) * grid_width
					h = np.square(n_h) * config.image_size / config.no_grid  #square (n_h) * grid_height

					x_cent = grid_col * (config.image_size / config.no_grid) + n_x_cent * (config.image_size / config.no_grid)
					y_cent = grid_row * (config.image_size / config.no_grid) + n_y_cent * (config.image_size / config.no_grid)

					#Converting mid point to upper left and lower right
					x1 = x_cent - w/2
					y1 = y_cent - h/2
					x2 = x_cent + w/2
					y2 = y_cent + h/2

					#Truncating between 0 and self.image_size
					xmin = max(min((float(x1) - 1), config.image_size - 1), 0)
					xmax = max(min((float(x2) - 1), config.image_size - 1), 0)
					ymin = max(min((float(y1) - 1), config.image_size - 1), 0)
					ymax = max(min((float(y2) - 1), config.image_size - 1), 0)
					best_box = [xmin, ymin, xmax, ymax]

					# best_box = [best_box[0] * (1/self.w_ratio), best_box[1] * (1/self.h_ratio), best_box[2] *(1/self.w_ratio), best_box[3] * (1/self.h_ratio)]
					
					all_best_boxes.append(best_box)
					best_scores.append(best_score)

					# print ("x1: {}, y1: {}, x2: {}, y2: {}".format(xmin, ymin, xmax, ymax))
					# print ("Best score: {}".format(best_score))
					# print ("Best class: {}".format(best_class))

			#Non maximal supression
			all_best_boxes = np.array(all_best_boxes).reshape(-1,4)
			sup_box_index = self.non_maximal_supression(all_best_boxes, best_scores)
			
			for box_id in sup_box_index:
				cur_box = all_best_boxes[box_id,:]
				image_ = cv2.rectangle(image_, (int(cur_box[0]),int(cur_box[1])),(int(cur_box[2]),int(cur_box[3])), (0,0,255), 2)

			ground_truth_boxes = np.array(ground_truth_boxes).reshape(-1,4)	

			for box_id in range(ground_truth_boxes.shape[0]):
				cur_box = ground_truth_boxes[box_id,:]
				image_ = cv2.rectangle(image_, (int(cur_box[0] - cur_box[2]/2.0), int(cur_box[1] - cur_box[3]/2.0)), (int(cur_box[0] + cur_box[2]/2.0), int(cur_box[1] + cur_box[3]/2.0)), (0,255,0), 2)

			image_ = cv2.normalize(image_.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
			image_ = image_.astype('float')

			# class_image[grid_row, grid_col] = best_class
			# class_image = cv2.normalize(class_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
			# class_image = class_image.astype('float')

			# viz = np.concatenate((image_, class_image), 1)	

			# viz = viz.astype('float')

			cv2.imshow("image", image_)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
