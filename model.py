import tensorflow as tf
import numpy as np
import config

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

	def conv_layers_typ1(self, x):
	
		#Repeat Convolutional layer 1
		x1 = tf.layers.conv2d(x, 256,1, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
		x1 = tf.nn.leaky_relu(x1, alpha)

		#Repeat convolutional layer 2
		x2 = tf.layers.conv2d(x1, 512, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
		x2 = tf.nn.leaky_relu(x2, alpha)

		return x2


	def conv_layers_typ2(self, x):
	
		#Repeat convolutional layer 1
		x1 = tf.layers.conv2d(x, 512, 1,'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
		x1 = tf.nn.leaky_relu(x1, alpha)

		#Repeat convolutional layer 2
		
		x2 = tf.layers.conv2d(x1, 1024, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
		x2 = tf.nn.leaky_relu(x2, alpha)

		return x2


	def yolo_model(self, alpha, dropout_rate, is_training, scope ="yolo_model"):
	
		with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
			
			#Convolutional layer 1
			x1 = tf.layers.conv2d(x,64,7,2,'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x1 = tf.nn.leaky_relu(x1, alpha)


			#Max pool layer 1
			x2 = tf.layers.max_pooling2d(x1,2,2)
			
			#Convolutional layer 2
			x3 = tf.layers.conv2d(x2, 192, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x3 = tf.nn.leaky_relu(x3, alpha)

			#Max pool layer 2
			x4 = tf.layers.max_pooling2d(x3,2,2)
			
			#Convolutional layer 3
			x5 = tf.layers.conv2d(x4, 128,1, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x5 = tf.nn.leaky_relu(x5, alpha)

			#Convolutional layer 4
			x6 = tf.layers.conv2d(x5, 256, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x6 = tf.nn.leaky_relu(x6, alpha)

			#Convolutional layer 5
			x7 = tf.layers.conv2d(x6, 256, 1, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x7 = tf.nn.leaky_relu(x7, alpha)

			#Convolutional layer 6
			x8 = tf.layers.conv2d(x7, 512, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x8 = tf.nn.leaky_relu(x8, alpha)

			#Max pool layer 3
			x9 = tf.layers.max_pooling2d(x8, 2,2)
			
			#Convolutional layers 7,8
			x10 = conv_layers_typ1(x9)

			#Convolutional layers 9,10
			x11 = conv_layers_typ1(x10)

			#Convolutional layers 11,12
			x12 = conv_layers_typ1(x11)

			#Convolutional layers 13,14
			x13 = conv_layers_typ1(x12)

			#Convolutional layer 15
			x14 = tf.layers.conv2d(x13, 512, 1, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x14 = tf.nn.leaky_relu(x14, alpha)

			#Convolutional layer 16
			x15 = tf.layers.conv2d(x14, 1024, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x15 = tf.nn.leaky_relu(x15, alpha)

			#Max pool layer 4
			x16 = tf.layers.max_pooling2d(x15, 2,2)
			
			#Convolutional layer 17, 18
			x17 = conv_layers_typ2(x16)
			
			#Convolutional layer 19, 20
			x18 = conv_layers_typ2(x17)
			
			#Convolutional layer 21
			x19 = tf.layers.conv2d(x18, 1024, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x19 = tf.nn.leaky_relu(x19, alpha)

			#Convolutional layer 22
			x20 = tf.layers.conv2d(x19, 1024, 3, 2, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x20 = tf.nn.leaky_relu(x20, alpha)

			#Convolutional layer 23
			x21 = tf.layers.conv2d(x20, 1024, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x21 = tf.nn.leaky_relu(x21, alpha)

			#Convolutional layer 24
			x22 = tf.layers.conv2d(x21, 1024, 3, 'same', kernel_intializer = tf.contrib.layers.xavier_initializer())
			x22 = tf.nn.leaky_relu(x22, alpha)

			#Fully connected layer 1
			flat1 = tf.contrib.layers.flatten(x22)
			x23 = tf.layers.dense(flat1, 512)
			x23 = tf.nn.leaky_relu(x23, alpha)
			# x23 = tf.layers.dropout(inputs=x23, rate= dropout_rate, training= is_training)


			#Fully connected layer 2
			flat2 = tf.contrib.layers.flatten(x23)
			x24 = tf.layers.dense(flat2, 4096)
			x24 = tf.nn.leaky_relu(x24, alpha)
			x24 = tf.layers.dropout(inputs=x24, rate= dropout_rate, training= is_training)
			
			#Reshaping the output of the last layer to the size (batch size, S,S,(5*B+C))
			out = tf.reshape(x24, (-1, self.no_grid,self.no_grid,5*self.no_boxes_per_cell + self.no_classes))
			
			return out


	def cal_iou(self, predict_boxes, label_boxes, scope = "iou"):
		
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

			boxes1 = tf.stack([predict_boxes[:,:,:,:,0],
							   predict_boxes[:,:,:,:,1],
							   (predict_boxes[:,:,:,:,0] + predict_boxes[:,:,:,:,2]),
							   (predict_boxes[:,:,:,:,1] + predict_boxes[:,:,:,:,3])])
			boxes1 = tf.transpose(boxes1, [1,2,3,4,0])
			
			#Calculating the actual label box upper left x,y and lower right x,y
			boxes2 = tf.stack([label_boxes[:,:,:,:,0],
							   label_boxes[:,:,:,:,1],
							   (label_boxes[:,:,:,:,0] + label_boxes[:,:,:,:,2]),
							   (label_boxes[:,:,:,:,1] + label_boxes[:,:,:,:,3])])
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
			return tf.clip_by_value(iArea / uArea . 0.0, 1.0)


	def loss(self, prediction, labels, scope = "loss_definition"):
	
		"""
		#Dimensions of labels:
		Labels will be of dimension [batch_size, no_grid, no_grid, 5 (1st dimension would be confidence (1 if object 
		 is present otherwise 0), 2nd and 3rd dimension would be upper x and y, 4th and 5th would be width and height)+ C (class probability)]
		 
		Labels are not of the shape [batch_size, no_grid, no_grid, 5*B+C] because for each bounding box predictor per cell
		dimensions of the object are same
		 
		#Dimensions of prediction:
		Prediction will be of dimension [batch_size, no_grid, no_grid, 5*B+C]
		in the 4th dimension: (1st B are confidence score, next 4*B are [x,y,w,h] repeated B times, and last C are class prob)
		""" 
		with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

			### extracting classes and boxes from labels
			label_classes = tf.reshape(labels[:,:,:,5:], [self.batch_size, self.no_grid, self.no_grid, self.no_classes])
			label_boxes = tf.reshape(labels[:,:,:,1:5], [self.batch_size, self.no_grid, self.no_grid, 1, 4])
			label_boxes = tf.tile(label_boxes, [1,1,1,self.boxes_per_cell,1]) / self.self.image_size
			
			### extracting classes, boxes and confidence from prediction
			predict_classes = tf.reshape(prediction[:,:,:,5*self.boxes_per_cell:], [self.batch_size, self.no_grid, self.no_grid, self.no_classes])
			predict_boxes = tf.reshape(prediction[:,:,:,self.boxes_per_cell:5*self.boxes_per_cell], [self.batch_size, self.no_grid, self.no_grid, self.boxes_per_cell, 4])
			predict_confidence = tf.reshape(prediction[:,:,:,:self.boxes_per_cell], [self.batch_size, self.no_grid, self.no_grid, self.boxes_per_cell])

			###Calculating offset for correction of prediction and formatting of labels
			offset = np.transpose(np.reshape(np.array([np.arange(self.no_grid)] * self.no_grid * self.boxes_per_cell), (self.boxes_per_cell, self.no_grid, self.no_grid)), (1, 2, 0))
			offset = tf.constant(offset, dtype = tf.float32)
			offset = tf.reshape(offset, [1, self.no_grid, self.no_grid, self.boxes_per_cell])
			offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
			
			###Correction and formatting
			#Correcting predict_boxes for calculating IOU
			corr_predict_boxes = tf.stack([(predict_boxes[:,:,:,:,0] + offset) / self.no_grid,
										  (predict_boxes[:,:,:,:,1] + np.transpose(offset, (0,2,1,3))) / self.no_grid,
										  tf.square(predict_boxes[:,:,:,:,2]),
										  tf.square(predict_boxes[:,:,:,:,3])])
			corr_predict_boxes = tf.transpose(corr_predict_boxes, [1,2,3,4,0]) #As tf.stack appends new dimension in the front
			
			#Formatting label boxes according by normalization and subtracting offset
			format_label_boxes = tf.stack([label_boxes[:,:,:,:,0] * self.no_grid - offset,
										   label_boxes[:,:,:,:,1] + np.transpose(offset, (0,2,1,3)) / self.no_grid,
										   tf.sqrt(label_boxes[:,:,:,:,2]),
										   tf.sqrt(label_boxes[:,:,:,:,3])])
			format_label_boxes = tf.transpose(format_label_boxes, [1,2,3,4,0]) #As tf.stack appends new dimension in the front
			
			#Calculated IOU for each box (batch_size, no_grid, no_grid, boxes_per_cell)
			calc_iou_boxes = calc_iou(corr_predict_boxes, label_boxes)
			
			###Calculating difference masks
			#Mask if the object is present in the cell or not (batch_size, no_grid, no_grid, 1)
			object_presence_map = tf.reshape(labels[:,:,:,0], [self.batch_size, self.no_grid, self.no_grid, 1]) #if object is present in the grid cell or not
			
			#Calculating object max (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell))
			iou_mask = tf.reduce_max(calc_iou_boxes, axis = 3, keepdims = True)
			iou_mask = tf.cast((calc_iou_boxes>=iou_mask), dtype=tf.float64)
			object_mask = iou_mask * object_presence_map
			
			#Calculating no object mask (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell))
			no_object_mask = tf.ones_like(object_mask, dtype=tf.float64) - object_mask
			
			#Calculating coordinate mask (float 1,0 of shape (batch_size, no_grid, no_grid, boxes_per_cell, 1))
			coord_mask = tf.expand_dims(object_mask, 4)
			
			
			### Losses
			#Class loss
			self.class_loss = self.class_scale * tf.reduce_mean(tf.reduce_sum(tf.square(object_presence_map*(predict_classes - label_classes)), axis = [1,2,3]))
			
			#Calculating confidence loss
			self.confidence_obj_loss = object_mask * (predict_confidence - calc_iou_boxes)  # we want to make confidence score same as iou when obj is present
			self.confidence_obj_loss = self.confidence_obj_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.confidence_obj_loss), axis = [1,2,3]))
			
			self.confidence_noobj_loss = no_object_mask * (predicted_confidence) #we want to make the confidence score 0 when obj is not present
			self.confidence_noobj_loss = self.confidence_noobj_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.confidence_noobj_loss), axis = [1,2,3]))
			
			#Calculating coordinates loss
			self.coord_loss = coord_mask * (predict_boxes - format_label_boxes)
			self.coord_loss = self.coord_scale * tf.reduce_mean(tf.reduce_sum(tf.square(self.coord_loss), axis = [1,2,3]))
			
			#Adding up the losses
			self.total_loss = self.class_loss + self.confidence_obj_loss + self.confidence_noobj_loss + self.coord_loss

			return self.total_loss, self.class_loss, self.confidence_obj_loss, self.confidence_noobj_loss, self.coord_loss

