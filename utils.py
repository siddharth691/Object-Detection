import os
import cv2
import numpy as np
import random
import config
import json	

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
			path = self.trainPath

		elif(phase == 'test'):

			imageId = self.testId
			path = self.testPath


		random.shuffle(imageId)
		label_dict = []
		for imgId in imageId:
			label, no_obj = self.createImageLabel(imgId)

			if(no_obj == 0):
				continue
			path = os.path.join(path, self.file['imgs'][imgId]['path'])
			Id = self.file['imgs'][imgId]['id']
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




