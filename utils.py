import os
import numpy as np
import random
import config_local as config
import json	
import cv2
import matplotlib.pyplot as plt
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

		random.shuffle(imageId)
		label_dict = []
		for imgId in imageId:
			label, no_obj = self.createImageLabel(imgId)

			if(no_obj == 0):
				continue
			
			Id = self.file['imgs'][imgId]['id']
			path = os.path.join(path_root, str(Id)+".jpg")
			label_dict.append({"image_path": path, "id": Id, "label": label})

		return label_dict



	def createImageLabel(self, imgId):

		objs = self.file['imgs'][imgId]['objects']
		label = np.zeros((config.no_grid, config.no_grid, 5 + self.no_classes))
		for obj in objs:

			xmin = max(min((float(obj['bbox']['xmin']) - 1) * self.w_ratio, self.image_size - 1), 0)
			xmax = max(min((float(obj['bbox']['xmax']) - 1) * self.w_ratio, self.image_size - 1), 0)
			ymin = max(min((float(obj['bbox']['ymin']) - 1) * self.h_ratio, self.image_size - 1), 0)
			ymax = max(min((float(obj['bbox']['ymax']) - 1) * self.h_ratio, self.image_size - 1), 0)

			boxes = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin]
			x_ind = int(boxes[0] * config.no_grid / self.image_size)
			y_ind = int(boxes[1] * config.no_grid / self.image_size)

			class_cat = obj['category']
			class_cat_id = self.classes_dict[class_cat]

			if(label[y_ind, x_ind, 0] == 1):
				continue

			label[y_ind,x_ind,0] = 1
			label[y_ind, x_ind,1:5] = boxes
			label[y_ind, x_ind,5+class_cat_id] = 1

		return label, len(objs)

	def display_bounding_box_batch(self, prediction, image_batch):

		fig, ax = plt.subplots()

		for batch in range(config.batch_size):

			predict_ = prediction[batch, :,:,:]
			image_ = image_batch[batch,:,:,:]

			class_image = np.zeros(image_.shape)
			for grid_row in range(config.no_grid):
				for grid_col in range(config.no_grid):

					boxes_with_prob = predict_[grid_row, grid_col, :]
					box_confidence = boxes_with_prob[:config.no_boxes_per_cell]
					best_box = np.argmax(box_confidence)


					boxes = boxes_with_prob[config.no_boxes_per_cell: 5 * config.no_boxes_per_cell]
					best_box_indexes = boxes[best_box * 4: (best_box + 1) * 4]
					best_box_indexes = [best_box_indexes[0] * (1/self.w_ratio), best_box_indexes[1] * (1/self.h_ratio), best_box_indexes[2] *(1/self.w_ratio), best_box_indexes[3] * (1/self.h_ratio)]

					class_confidence = boxes_with_prob[5 * config.no_boxes_per_cell :]
					best_class = np.argmax(class_confidence)

					x1 = best_box_indexes[0] - best_box_indexes[2]/2
					y1 = best_box_indexes[1] - best_box_indexes[3]/2
					x2 = best_box_indexes[0] + best_box_indexes[2]/2
					y2 = best_box_indexes[1] + best_box_indexes[3]/2

					image_ = cv2.rectangle(image_, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,0), 1)
					
					class_image[grid_row, grid_col] = best_class

			class_image = class_image.astype('float')

			fig.canvas.mpl_connect('key_press_event', self.press)
			plt.subplot(1,2,1)
			plt.imshow(image_)
			plt.axis('tight')

			plt.subplot(1,2,2)
			plt.imshow(class_image)
			plt.axis('tight')
			plt.show()
			input("Press Enter to continue...")
			
			getch.getch()
			plt.close()







		