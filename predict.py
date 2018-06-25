import os
import tensorflow as tf
import config_local as config
import model
from utils import readData
import time
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
import matplotlib.pyplot as plt
import cv2

# config.batch_size = 100
checkpoint_path = os.path.join(config.checkpoint_path, "yolo_last_layer.ckpt")
model_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/yolo.ckpt"
log_file = os.path.join(config.log_file_path ,'status_log.txt')

#Changing parameters to values in yolo.ckpt for loading


model = model.model()
utils = readData(config.dir_path)
label_list = utils.createLabels('test')

sess = tf.InteractiveSession()
saver_pretrained = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''))
sess.run(tf.global_variables_initializer())
saver_pretrained.restore(sess, model_path)


prediction = model.yolo(config.dropout)
model.loss()

saver_last_layer = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'last_layer'))

#Initialize uninitialized variables
uninitialized_vars = []
for var in tf.global_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)

init_new_vars_op = tf.initialize_variables(uninitialized_vars)
sess.run(init_new_vars_op)

#Restoring last layer variables
saver_last_layer.restore(sess, checkpoint_path)
no_images = len([img for img in os.listdir(utils.testPath) if img.endswith('.jpg')])

total_loss = 0
loss_test = []
no_batches = 0
for x in range(0, no_images - config.batch_size, config.batch_size):

	print("current batch no: {}".format(no_batches))
	images = np.zeros((config.batch_size, config.image_size, config.image_size, 3))
	label = np.zeros((config.batch_size, config.no_grid, config.no_grid, 5 + config.no_classes))

	for batch in range(config.batch_size):

		image_path = label_list[x + batch]['image_path']
		images[batch, :, :, :] = utils.read_img(image_path)
		label[batch, :, :, :] = label_list[x + batch]['label']

	loss, predict = sess.run([model.total_loss, prediction], feed_dict = {model.images: images, model.labels: label})

	#Plotting images
	utils.display_bounding_box_batch(predict, images)


	print ("loss : {}".format(loss))
	loss_test.append(loss)

	total_loss += loss

	no_batches+=1


print ("total_loss = {}".format(total_loss))


fig,ax = plt.subplots()
ax.plot(range(no_batches), total_loss, linewidth =2)
plt.axis('tight')
plt.xlabel('Batch number')
plt.ylabel('Total loss for that batch')
plt.title('Test Loss')
plt.show()