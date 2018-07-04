import tensorflow as tf
import numpy as np
import config
from functools import *

class model:

	def __init__(self):

		self.batch_size = config.batch_size
		self.image_width = config.image_width
		self.image_height = config.image_height
		self.image_size = config.image_size

		self.no_grid = config.no_grid
		self.no_classes = config.no_classes
		self.no_boxes_per_cell = config.no_boxes_per_cell

		#Scales
		self.class_scale = config.class_scale
		self.confidence_obj_scale = config.confidence_obj_scale
		self.confidence_noobj_scale = config.confidence_noobj_scale
		self.coord_scale = config.coord_scale


		self.images = tf.placeholder(tf.float32, shape = [config.batch_size, config.image_size,config.image_size,3], name='image_placeholder')
		self.labels = tf.placeholder(tf.float32, shape = [config.batch_size, config.no_grid, config.no_grid, 5+config.no_classes], name='label_placeholder')
			
		#Calling the model to define the yolo model
		self.prediction = self.yolo_model(dropout_rate = config.dropout)
		


	def create_connected_layer(self, input_layer, d0, leaky, weight_index, name = 'Variable', pretraining = False):
		weight_shape = [int(input_layer.get_shape()[1]), d0]
		bias_shape = [d0]


		if(pretraining ==  False):

			with tf.variable_scope("", reuse = tf.AUTO_REUSE):

				weight_name = name +'_'+ str(weight_index)
				bias_name = name +'_'+ str(weight_index + 1)
				weight = tf.get_variable('%s' % (weight_name) , weight_shape, initializer=tf.contrib.layers.xavier_initializer())
				bias = tf.get_variable('%s' % (bias_name) , bias_shape, initializer=tf.constant_initializer(0.0))
		
		else:
			with tf.variable_scope("last_layer", reuse = tf.AUTO_REUSE):
				weight_name = name +'_'+ str(weight_index)
				bias_name = name +'_'+ str(weight_index + 1)
				weight = tf.get_variable('%s' % (weight_name) , weight_shape, initializer=tf.contrib.layers.xavier_initializer())
				bias = tf.get_variable('%s' % (bias_name) , bias_shape, initializer=tf.constant_initializer(0.0))

		return self.activation(tf.add(tf.matmul(input_layer, weight), bias), leaky)


	def activation(self, input_layer, leaky = True, pretraining = False):


		if leaky:
			return tf.maximum(input_layer, tf.scalar_mul(config.alpha, input_layer))
		else:
			return input_layer

	def create_maxpool_layer(self, input_layer, d0, d1, stride):
		return tf.nn.max_pool(input_layer, ksize = [1, d0, d1, 1], strides = [1, stride, stride, 1], padding = 'SAME')

	def create_dropout_layer(self, input_layer, prob):
		return tf.nn.dropout(input_layer, prob)

	def create_conv_layer(self, input_layer, d0, d1, filters, stride, index, scope = 'Variable', pretraining = False):

		channels = int(input_layer.get_shape()[3])
		weight_shape = [d0, d1, channels, filters]
		bias_shape = [filters]

		if (pretraining == False):

			if(index == 0):
				weight_name = scope
			else:
				weight_name =scope +'_'+ str(index)
			bias_name = scope +'_'+ str(index + 1)

			with tf.variable_scope("", reuse = tf.AUTO_REUSE):

				weight = tf.get_variable( '%s' % (weight_name), weight_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
				bias = tf.get_variable( '%s' % (bias_name), bias_shape, initializer=tf.constant_initializer(0.0))

		else:
			with tf.variable_scope("last_layer", reuse = tf.AUTO_REUSE):
				weight_name =scope +'_'+ str(index)
				bias_name = scope +'_'+ str(index + 1)

				weight = tf.get_variable( '%s' % (weight_name), weight_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
				bias = tf.get_variable( '%s' % (bias_name), bias_shape, initializer=tf.constant_initializer(0.0))


		d0_pad = int(d0/2)
		d1_pad = int(d1/2)
		input_layer_padded = tf.pad(input_layer, paddings = [[0, 0], [d0_pad, d0_pad], [d1_pad, d1_pad], [0, 0]])

		
		# we need VALID paddings here to match the sizing calculation for output of convolutional used by darknet
		convolution = tf.nn.conv2d(input = input_layer_padded, filter = weight, strides = [1, stride, stride, 1], padding='VALID')
		convolution_bias = tf.add(convolution, bias)

		return self.activation(convolution_bias, pretraining = pretraining)


	def yolo(self, dropout_rate):

		# conv_layer26 = self.yolo_model(dropout_rate, feature_extract = True)
		# conv_layer26 = tf.stop_gradient(conv_layer26)
		# conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1,46,pretraining = True)
		# # flatten layer for connection to fully connected layer
		# conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
		# conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])
		# connected_layer28 = self.create_connected_layer(conv_layer27_flatten, 512, True, 48,pretraining = True)
		# connected_layer29 = self.create_connected_layer(connected_layer28, 4096, True, 50,pretraining = True)
		# dropout_layer30 = self.create_dropout_layer(connected_layer29, dropout_rate)
		# connected_layer31 = self.create_connected_layer(dropout_layer30, self.no_grid*self.no_grid*(5*self.no_boxes_per_cell + self.no_classes), False, 52,pretraining = True)

		# self.prediction = tf.reshape(connected_layer31, (self.batch_size, self.no_grid,self.no_grid,5*self.no_boxes_per_cell + self.no_classes))

		conv_layer10 = self.yolo_model(dropout_rate, feature_extract = True)
		conv_layer10 = tf.stop_gradient(conv_layer10)
		conv_layer11 = self.create_conv_layer(conv_layer10, 1, 1, 256, 1, 16, pretraining = True)
		conv_layer12 = self.create_conv_layer(conv_layer11, 3, 3, 512, 1, 18, pretraining = True)
		conv_layer13 = self.create_conv_layer(conv_layer12, 1, 1, 256, 1, 20, pretraining = True)
		conv_layer14 = self.create_conv_layer(conv_layer13, 3, 3, 512, 1, 22, pretraining = True)
		conv_layer15 = self.create_conv_layer(conv_layer14, 1, 1, 256, 1, 24, pretraining = True)
		conv_layer16 = self.create_conv_layer(conv_layer15, 3, 3, 512, 1, 26, pretraining = True)
		conv_layer17 = self.create_conv_layer(conv_layer16, 1, 1, 512, 1, 28, pretraining = True)
		conv_layer18 = self.create_conv_layer(conv_layer17, 3, 3, 1024, 1,30, pretraining = True)
		maxpool_layer19 = self.create_maxpool_layer(conv_layer18, 2, 2, 2)
		conv_layer20 = self.create_conv_layer(maxpool_layer19, 1, 1, 512, 1 ,32, pretraining = True)
		conv_layer21 = self.create_conv_layer(conv_layer20, 3, 3, 1024, 1, 34, pretraining = True)
		conv_layer22 = self.create_conv_layer(conv_layer21, 1, 1, 512, 1, 36, pretraining = True)
		conv_layer23 = self.create_conv_layer(conv_layer22, 3, 3, 1024, 1, 38, pretraining = True)
		conv_layer24 = self.create_conv_layer(conv_layer23, 3, 3, 1024, 1, 40, pretraining = True)
		conv_layer25 = self.create_conv_layer(conv_layer24,  3, 3, 1024, 2, 42, pretraining = True)
		conv_layer26 = self.create_conv_layer(conv_layer25, 3, 3, 1024, 1, 44, pretraining = True)
		conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1, 46, pretraining = True)
		# flatten layer for connection to fully connected layer
		conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
		conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])
		connected_layer28 = self.create_connected_layer(conv_layer27_flatten, 512, True, 48,pretraining = True)
		connected_layer29 = self.create_connected_layer(connected_layer28, 4096, True, 50,pretraining = True)
		dropout_layer30 = self.create_dropout_layer(connected_layer29, dropout_rate)
		connected_layer31 = self.create_connected_layer(dropout_layer30, self.no_grid*self.no_grid*(5*self.no_boxes_per_cell + self.no_classes), False, 52,pretraining = True)

		self.prediction = tf.reshape(connected_layer31, (self.batch_size, self.no_grid,self.no_grid,5*self.no_boxes_per_cell + self.no_classes))


		return self.prediction

	def yolo_model(self, dropout_rate, feature_extract= False):

		if(feature_extract == False):
			images = tf.image.resize_images(self.images, tf.constant([448, 448]))
			conv_layer0 = self.create_conv_layer(images, 7, 7, 64, 2,0)

		else:
			conv_layer0 = self.create_conv_layer(self.images, 7,7,64,2,0)


		maxpool_layer1 = self.create_maxpool_layer(conv_layer0, 2, 2, 2)
		conv_layer2 = self.create_conv_layer(maxpool_layer1, 3, 3, 192, 1,2)
		maxpool_layer3 = self.create_maxpool_layer(conv_layer2, 2, 2, 2 )
		conv_layer4 = self.create_conv_layer(maxpool_layer3, 1, 1, 128, 1,4)
		conv_layer5 = self.create_conv_layer(conv_layer4, 3, 3, 256, 1,6)
		conv_layer6 = self.create_conv_layer(conv_layer5, 1, 1, 256, 1,8)
		conv_layer7 = self.create_conv_layer(conv_layer6, 3, 3, 512, 1,10)
		maxpool_layer8 = self.create_maxpool_layer(conv_layer7, 2, 2, 2)
		conv_layer9 = self.create_conv_layer(maxpool_layer8, 1, 1, 256, 1,12)
		conv_layer10 = self.create_conv_layer(conv_layer9, 3, 3, 512, 1,14)
		#Stopping till layer before this
		if (feature_extract):
			return conv_layer10
		conv_layer11 = self.create_conv_layer(conv_layer10, 1, 1, 256, 1,16)
		conv_layer12 = self.create_conv_layer(conv_layer11, 3, 3, 512, 1,18)
		conv_layer13 = self.create_conv_layer(conv_layer12, 1, 1, 256, 1,20)
		conv_layer14 = self.create_conv_layer(conv_layer13, 3, 3, 512, 1,22)
		conv_layer15 = self.create_conv_layer(conv_layer14, 1, 1, 256, 1,24)
		conv_layer16 = self.create_conv_layer(conv_layer15, 3, 3, 512, 1,26)
		conv_layer17 = self.create_conv_layer(conv_layer16, 1, 1, 512, 1,28)
		conv_layer18 = self.create_conv_layer(conv_layer17, 3, 3, 1024, 1,30)
		maxpool_layer19 = self.create_maxpool_layer(conv_layer18, 2, 2, 2)
		conv_layer20 = self.create_conv_layer(maxpool_layer19, 1, 1, 512, 1,32)
		conv_layer21 = self.create_conv_layer(conv_layer20, 3, 3, 1024, 1,34)
		conv_layer22 = self.create_conv_layer(conv_layer21, 1, 1, 512, 1,36)
		conv_layer23 = self.create_conv_layer(conv_layer22, 3, 3, 1024, 1,38)
		conv_layer24 = self.create_conv_layer(conv_layer23, 3, 3, 1024, 1,40)
		conv_layer25 = self.create_conv_layer(conv_layer24, 3, 3, 1024, 2,42)
		conv_layer26 = self.create_conv_layer(conv_layer25, 3, 3, 1024, 1,44)
		conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1,46)
		
		# flatten layer for connection to fully connected layer
		conv_layer27_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer27.get_shape()[1:]))
		conv_layer27_flatten = tf.reshape(tf.transpose(conv_layer27, (0, 3, 1, 2)), [-1, conv_layer27_flatten_dim])

		connected_layer28 = self.create_connected_layer(conv_layer27_flatten, 512, True, 48)

		
		connected_layer29 = self.create_connected_layer(connected_layer28, 4096, True, 50)

		dropout_layer30 = self.create_dropout_layer(connected_layer29, dropout_rate)
		connected_layer31 = self.create_connected_layer(dropout_layer30, 1470, False, 52)

		self.prediction = tf.reshape(connected_layer31, (self.batch_size, 7,7,5*2 + 20))

		return self.prediction

	# def yolo_model(self, alpha, dropout_rate, is_training, scope = "yolo_model"):
	
	# 	with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
				
	# 		#Convolutional layer 1
	# 		x1 = tf.layers.conv2d(self.images,64,7,2,'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x1 = tf.nn.leaky_relu(x1, alpha)
	# 		#Max pool layer 1
	# 		x2 = tf.layers.max_pooling2d(x1,2,2)
			
	# 		#Convolutional layer 2
	# 		x3 = tf.layers.conv2d(x2, 192, 3, 1,'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x3 = tf.nn.leaky_relu(x3, alpha)

	# 		#Max pool layer 2
	# 		x4 = tf.layers.max_pooling2d(x3,2,2)
			
	# 		#Convolutional layer 3
	# 		x5 = tf.layers.conv2d(x4, 128,1,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x5 = tf.nn.leaky_relu(x5, alpha)

	# 		#Convolutional layer 4
	# 		x6 = tf.layers.conv2d(x5, 256, 3,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x6 = tf.nn.leaky_relu(x6, alpha)

	# 		#Convolutional layer 5
	# 		x7 = tf.layers.conv2d(x6, 256, 1,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x7 = tf.nn.leaky_relu(x7, alpha)

	# 		#Convolutional layer 6
	# 		x8 = tf.layers.conv2d(x7, 512, 3,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x8 = tf.nn.leaky_relu(x8, alpha)

	# 		#Max pool layer 3
	# 		x9 = tf.layers.max_pooling2d(x8, 2,2)
			
	# 		#Convolutional layers 7,8
	# 		x10 = self.conv_layers_typ1(x9, alpha)

	# 		#Convolutional layers 9,10
	# 		x11 = self.conv_layers_typ1(x10, alpha)

	# 		#Convolutional layers 11,12
	# 		x12 = self.conv_layers_typ1(x11, alpha)

	# 		#Convolutional layers 13,14
	# 		x13 = self.conv_layers_typ1(x12, alpha)

	# 		#Convolutional layer 15
	# 		x14 = tf.layers.conv2d(x13, 512, 1,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x14 = tf.nn.leaky_relu(x14, alpha)

	# 		#Convolutional layer 16
	# 		x15 = tf.layers.conv2d(x14, 1024, 3,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x15 = tf.nn.leaky_relu(x15, alpha)

	# 		#Max pool layer 4
	# 		x16 = tf.layers.max_pooling2d(x15, 2,2)
			
	# 		#Convolutional layer 17, 18
	# 		x17 = self.conv_layers_typ2(x16, alpha)
			
	# 		#Convolutional layer 19, 20
	# 		x18 = self.conv_layers_typ2(x17, alpha)
			
	# 		#Convolutional layer 21
	# 		x19 = tf.layers.conv2d(x18, 1024, 3,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x19 = tf.nn.leaky_relu(x19, alpha)

	# 		#Convolutional layer 22
	# 		x20 = tf.layers.conv2d(x19, 1024, 3, 2, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x20 = tf.nn.leaky_relu(x20, alpha)

	# 		#Convolutional layer 23
	# 		x21 = tf.layers.conv2d(x20, 1024, 3, 1,'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x21 = tf.nn.leaky_relu(x21, alpha)

	# 		#Convolutional layer 24
	# 		x22 = tf.layers.conv2d(x21, 1024, 3,1, 'same', kernel_initializer = tf.contrib.layers.xavier_initializer())
	# 		x22 = tf.nn.leaky_relu(x22, alpha)

	# 		#Fully connected layer 1
	# 		flat1 = tf.contrib.layers.flatten(x22)
	# 		x23 = tf.layers.dense(flat1, 512)
	# 		x23 = tf.nn.leaky_relu(x23, alpha)
	# 		# x23 = tf.layers.dropout(inputs=x23, rate= dropout_rate, training= is_training)

	# 		#Fully connected layer 2
	# 		flat2 = tf.contrib.layers.flatten(x23)
	# 		x24 = tf.layers.dense(flat2, self.no_grid*self.no_grid*(5*self.no_boxes_per_cell+self.no_classes))
	# 		x24 = tf.nn.leaky_relu(x24, alpha)
	# 		x24 = tf.layers.dropout(inputs=x24, rate= dropout_rate, training= is_training)
			
	# 		#Reshaping the output of the last layer to the size (batch size, S,S,(5*B+C))
	# 		self.prediction = tf.reshape(x24, (self.batch_size, self.no_grid,self.no_grid,5*self.no_boxes_per_cell + self.no_classes))
			
	# 		return self.prediction


	def calc_iou(self, predict_boxes, label_boxes, scope = "iou"):
		
		"""
		Calculates IOU in multidimension
		
		Input:
		------
		predict_boxes (batch_size, no_grid, no_grid, boxes_per_cell, 4): predicted boxes (corrected)
		label_boxes (batch_size, no_grid, no_grid, boxes_per_cell, 4): label boxes
		
		Returns:
		--------
		IOU (batch_size, no_grid, no_grid, boxes_per_cell) : multidimensional IOU (truncated between 0 and 1)
		"""
		#Calculating the predicted box upper left x,y and lower right x,y

		with tf.variable_scope(scope, reuse= tf.AUTO_REUSE):

			boxes1 = tf.stack([predict_boxes[:, :, :, :, 0] - predict_boxes[:, :, :, :, 2] / 2.0,
							   predict_boxes[:, :, :, :, 1] - predict_boxes[:, :, :, :, 3] / 2.0,
							   predict_boxes[:, :, :, :, 0] + predict_boxes[:, :, :, :, 2] / 2.0,
							   predict_boxes[:, :, :, :, 1] + predict_boxes[:, :, :, :, 3] / 2.0])
			boxes1 = tf.transpose(boxes1, [1,2,3,4,0])
			
			#Calculating the actual label box upper left x,y and lower right x,y
			boxes2 = tf.stack([label_boxes[:, :, :, :, 0] - label_boxes[:, :, :, :, 2] / 2.0,
							   label_boxes[:, :, :, :, 1] - label_boxes[:, :, :, :, 3] / 2.0,
							   label_boxes[:, :, :, :, 0] + label_boxes[:, :, :, :, 2] / 2.0,
							   label_boxes[:, :, :, :, 1] + label_boxes[:, :, :, :, 3] / 2.0])
			boxes2 = tf.transpose(boxes2, [1,2,3,4,0])
			
			#Calculating the intersection box upper left x,y and lower right x,y
			ul = tf.maximum(boxes1[:,:,:,:,:2], boxes2[:,:,:,:,:2])
			lr = tf.minimum(boxes1[:,:,:,:,2:], boxes2[:,:,:,:,2:])
			
			#Calculating the area of the intersection box
			idiff = tf.maximum(0.0, lr - ul)
			iArea = idiff[:,:,:,:,0] * idiff[:,:,:,:,1]
			
			#Calculating the area of label box and predicted box
			lArea = (boxes2[:,:,:,:,2] - boxes2[:,:,:,:,0])*(boxes2[:,:,:,:,3] - boxes2[:,:,:,:,1])
			pArea = (boxes1[:,:,:,:,2] - boxes1[:,:,:,:,0])*(boxes1[:,:,:,:,3] - boxes1[:,:,:,:,1])
			
			
			#Calculating union area
			uArea = tf.maximum(lArea + pArea - iArea, 1e-10)
		
			#Clipping and returning the IOU
			return tf.clip_by_value(iArea / uArea , 0.0, 1.0)


	def loss(self, scope = "loss_definition"):
	
		"""
		#Dimensions of labels:
		Labels will be of dimension [batch_size, no_grid, no_grid, 5 (1st dimension would be confidence (1 if object 
		 is present otherwise 0), 2nd and 3rd dimension would be upper x and y, 4th and 5th would be width and height)+ C (class probability)]
		 
		Labels are not of the shape [batch_size, no_grid, no_grid, 5*B+C] because for each bounding box predictor per cell
		dimensions of the object are same
		 
		#Dimensions of prediction:
		Prediction will be of dimension [batch_size, no_grid, no_grid, 5*B+C]
		in the 4th dimension: (1st B are confidence score, next 4*B are [x,y,w,h] repeated B times, and last C are class prob)


		Prediction boxes xmid and ymid are offset from upper left coordinate of grid, and normalized by grid width and height
		Prediction width and height are square root of normalized (with grid width and height) actual bounding box width and height

		label boxes xmid and ymid are in image coordinates from upper left of the image
		label boxes width and height are actual bounding box width and height with respect to image

		""" 
		with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):


			### extracting classes and boxes from labels
			label_classes = tf.reshape(self.labels[:,:,:,5:], [self.batch_size, self.no_grid, self.no_grid, self.no_classes])
			label_boxes = tf.reshape(self.labels[:,:,:,1:5], [self.batch_size, self.no_grid, self.no_grid, 1, 4])
			label_boxes = tf.tile(label_boxes, [1,1,1,self.no_boxes_per_cell,1]) / self.image_size
			
			### extracting classes, boxes and confidence from prediction
			predict_classes = tf.reshape(self.prediction[:,:,:,5*self.no_boxes_per_cell:], [self.batch_size, self.no_grid, self.no_grid, self.no_classes])
			predict_boxes = tf.reshape(self.prediction[:,:,:,self.no_boxes_per_cell:5*self.no_boxes_per_cell], [self.batch_size, self.no_grid, self.no_grid, self.no_boxes_per_cell, 4])
			predict_confidence = tf.reshape(self.prediction[:,:,:,:self.no_boxes_per_cell], [self.batch_size, self.no_grid, self.no_grid, self.no_boxes_per_cell])

			###Calculating offset for correction of prediction and formatting of labels
			offset = np.transpose(np.reshape(np.array([np.arange(self.no_grid)] * self.no_grid * self.no_boxes_per_cell), (self.no_boxes_per_cell, self.no_grid, self.no_grid)), (1, 2, 0))
			offset = tf.constant(offset, dtype = tf.float32)
			offset = tf.reshape(offset, [1, self.no_grid, self.no_grid, self.no_boxes_per_cell])
			offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
			
			###Correction and formatting
			#Correcting predict_boxes for calculating IOU
			self.grid_width = self.image_size / self.no_grid
			self.grid_height = self.image_size / self.no_grid


			corr_predict_boxes = tf.stack([(predict_boxes[:,:,:,:,0] + offset) / self.no_grid,
										  (predict_boxes[:,:,:,:,1] + tf.transpose(offset, (0,2,1,3))) / self.no_grid,
										  tf.square(predict_boxes[:,:,:,:,2]) * self.grid_width / self.image_size,
										  tf.square(predict_boxes[:,:,:,:,3]) * self.grid_height / self.image_size])
			corr_predict_boxes = tf.transpose(corr_predict_boxes, [1,2,3,4,0]) #As tf.stack appends new dimension in the front
			
			

			#Formatting label boxes according by normalization and subtracting offset
			format_label_boxes = tf.stack([label_boxes[:,:,:,:,0] * self.no_grid - offset,
										   label_boxes[:,:,:,:,1] * self.no_grid - tf.transpose(offset, (0,2,1,3)),
										   tf.sqrt(label_boxes[:,:,:,:,2] * self.image_size / self.grid_width),
										   tf.sqrt(label_boxes[:,:,:,:,3] * self.image_size / self.grid_height)])
			format_label_boxes = tf.transpose(format_label_boxes, [1,2,3,4,0]) #As tf.stack appends new dimension in the front
			
			#Calculated IOU for each box (batch_size, no_grid, no_grid, boxes_per_cell)
			calc_iou_boxes = self.calc_iou(corr_predict_boxes, label_boxes)
			
			###Calculating difference masks
			#Mask if the object is present in the cell or not (batch_size, no_grid, no_grid, 1)
			object_presence_map = tf.reshape(self.labels[:,:,:,0], [self.batch_size, self.no_grid, self.no_grid, 1]) #if object is present in the grid cell or not
			
			#Calculating object max (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell))
			iou_mask = tf.reduce_max(calc_iou_boxes, axis = 3, keepdims = True)
			iou_mask = tf.cast((calc_iou_boxes>=iou_mask), dtype=tf.float32)
			object_mask = iou_mask * object_presence_map
			
			#Calculating no object mask (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell))
			no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
			
			#Calculating coordinate mask (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell, 1))
			coord_mask = tf.expand_dims(object_mask, 4)
			
			
			### Losses
			#Class loss
			self.class_loss = self.class_scale * tf.reduce_mean(tf.reduce_sum(tf.square(object_presence_map*(predict_classes - label_classes)), axis = [1,2,3]))
			
			#Calculating confidence loss
			self.confidence_obj_loss = object_mask * (predict_confidence - calc_iou_boxes)  # we want to make confidence score same as iou when obj is present
			self.confidence_obj_loss = self.confidence_obj_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.confidence_obj_loss), axis = [1,2,3]))
			
			self.confidence_noobj_loss = no_object_mask * (predict_confidence) #we want to make the confidence score 0 when obj is not present
			self.confidence_noobj_loss = self.confidence_noobj_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.confidence_noobj_loss), axis = [1,2,3]))
			
			#Calculating coordinates loss
			self.coord_loss = coord_mask * (predict_boxes - format_label_boxes)
			self.coord_loss = self.coord_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.coord_loss), axis = [1,2,3]))
			
		
			#Adding up the losses
			self.total_loss = self.class_loss + self.confidence_obj_loss + self.confidence_noobj_loss + self.coord_loss

			return self.total_loss