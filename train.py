import os
import tensorflow as tf
import config
import model
from utils import readData
import time
import numpy as np

sess = tf.InteractiveSession()
model = model.model()
utils = readData(config.dir_path)
label_list = utils.createLabels('train')

checkpoint_path = os.path.join(config.checkpoint_path, "model.ckpt")
log_file = os.path.join(config.log_file_path ,'status_log.txt')
no_images = int(config.dataset_ratio * len(label_list))

if(no_images <= config.batch_size):
	raise ValueError("dataset_ratio small for current batch size")


saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "yolo_model"))
sess.run(tf.global_variables_initializer())

#Restoring last checkpoint or pretrained model
# try:
# 	saver.restore(sess, checkpoint_path)
# 	print ('Loading from past checkpoint...')
# except Exception as e:
# 	print("Start training a fresh model")
# 	sess.run(tf.global_variables_initializer())



for epoch_no in range(config.epoch):
		
	last_time = time.time()
	total_loss = 0


	batch_no = 0
	
	for x in range(0, no_images - config.batch_size, config.batch_size):
		
		
		
		images = np.zeros((config.batch_size, config.image_size, config.image_size, 3))
		label = np.zeros((config.batch_size, config.no_grid, config.no_grid, 5 + config.no_classes))
		
		for batch in range(config.batch_size):
			image_path = label_list[x + batch]['image_path']
			images[batch, :, :, :] = utils.read_img(image_path)
			label[batch, :, :, :] = label_list[x + batch]['label']

		loss, _ = sess.run([model.total_loss, model.optimizer], feed_dict = {model.images: images, model.labels: label})
		total_loss += loss

		print("Current Batch Number: {}, loss: {}".format(batch_no, loss))
		if ((x + 1) % config.checkpoint == 0):
			print ("checkpoint reached: {}".format(str(x + 1)))
			
			checkpoint_status = "epoch: " + str(epoch_no + 1) +" , chkpt no: " + str(x+1) +" , loss: " + str(loss / (len(label_list) - config.batch_size / (config.batch_size * 1.0))) +  " ,  time (s) / epoch: " + str(time.time() - last_time)
			with open(log_file, "a") as myfile:
				myfile.write(checkpoint_status)
				myfile.write("\n")

			myfile.close()

		batch_no+=1

	np.random.shuffle(label_list)
	current_status ="epoch: " + str(epoch_no + 1) +" , loss: " + str(loss / (len(label_list) - config.batch_size / (config.batch_size * 1.0)))  + " ,  time (s) / epoch: " + str(time.time() - last_time)
	print (current_status)

	
	with open(log_file, "a") as myfile:
		myfile.write(current_status)
		myfile.write("\n")

	myfile.close()

	
	saver.save(sess, checkpoint_path)