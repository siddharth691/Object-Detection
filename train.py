import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
import config as config
import model
from utils import readData
import time
import numpy as np

checkpoint_path = os.path.join(config.checkpoint_path, "yolo_last_layer.ckpt")
model_path = config.pretrained_model_path
log_file = os.path.join(config.log_file_path ,'status_log.txt')

#Changing parameters to values in yolo.ckpt for loading
model = model.model()
utils = readData(config.dir_path)
label_list = utils.createLabels('train')

sess = tf.InteractiveSession()
# train_writer = tf.summary.FileWriter(config.log_file_path, sess.graph)
saver_pretrained = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''))
sess.run(tf.global_variables_initializer())
saver_pretrained.restore(sess, model_path)


prediction = model.yolo(config.dropout)
loss = model.loss()

#Add loss to scalar summary
# tf.summary.scalar("Total loss", loss)
#Add predictions to histogram
# tf.summary.histogram("predictions", prediction)

saver_last_layer = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'last_layer'))

global_batch_no = tf.Variable(0, name= 'global_step', trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = config.learning_rate, epsilon = 1e-10).minimize(model.total_loss, global_step = global_batch_no)

#Initialize uninitialized variables

uninitialized_vars = []
for var in tf.global_variables():
    try:
        sess.run(var)
    except tf.errors.FailedPreconditionError:
        uninitialized_vars.append(var)

init_new_vars_op = tf.initialize_variables(uninitialized_vars)
sess.run(init_new_vars_op)


no_images = int(config.dataset_ratio * len(label_list))
if(no_images <= config.batch_size):
	raise ValueError("dataset_ratio small for current batch size")


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

		# merge = tf.summary.merge_all()
		# loss = sess.run([model.total_loss], feed_dict = {model.images: images, model.labels: label})
		loss, _ = sess.run([model.total_loss, optimizer], feed_dict = {model.images: images, model.labels: label})
		# summary, loss, _ = sess.run([merge, model.total_loss, optimizer], feed_dict = {model.images: images, model.labels: label})


		print("Current Batch Number: {}, loss: {}".format(batch_no, loss))
		if ((x + 1) % config.checkpoint == 0):
			print ("checkpoint reached: {}".format(str(x + 1)))
			
			checkpoint_status = "epoch: " + str(epoch_no + 1) +" , chkpt no: " + str(x+1) +" , loss: " + str(loss / (len(label_list) - config.batch_size / (config.batch_size * 1.0))) +  " ,  time (s) / epoch: " + str(time.time() - last_time)
			with open(log_file, "a") as myfile:
				myfile.write(checkpoint_status)
				myfile.write("\n")

			myfile.close()

		batch_no+=1

	# train_writer.add_summary(summary, epoch_no)
	# total_loss+= loss[0]

	np.random.shuffle(label_list)
	current_status ="epoch: " + str(epoch_no + 1) +" , loss: " + str(loss / (len(label_list) - config.batch_size / (config.batch_size * 1.0)))  + " ,  time (s) / epoch: " + str(time.time() - last_time)
	print (current_status)

	loss_write = str(loss) +','
	with open(log_file, "a") as myfile:
		myfile.write(loss_write)
		myfile.write("\n")

	myfile.close()

	
	saver_last_layer.save(sess, checkpoint_path)