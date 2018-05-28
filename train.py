import os
import tensorflow as tf
import config
import model
from utils import readData

model = model.model()
utils = readData(config.dir_path)
label_list = utils.createLabels('train')

sess = tf.InteractiveSession()
saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "yolo_model"))
sess.run(tf.global_variables_initializer())


image_placeholder = tf.placeholder(tf.float32, shape = [self.batch_size,self.image_size,self.image_size,3], name='image_placeholder')
label_placeholder = tf.placeholder(tf.float32, shape = [self.batch_size, self.no_grid, self.no_grid, 5+self.no_classes], name='label_placeholder')
global_batch_no = tf.Variable(0, name= 'global_step', trainable = False)
checkpoint_path = os.path.join(config.checkpoint_path, "model.ckpt")
log_file = os.path.join(config.log_file_path ,'status_log.txt')

#Calling the model to define the yolo model
predictions = model.yolo_model()

#Calling loss function to define all the losses
total_loss_placeholder = model.loss(predictions, label_placeholder)

#Optimizer
optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(total_loss_placeholder, global_step = global_batch_no)


#Restoring last checkpoint or pretrained model
try:
    saver.restore(sess, checkpoint_path)
    print ('Loading from past checkpoint...')
except Exception as e:
    print 'Exit, atleast need a pretrained model'
    exit(0)



for epoch_no in range(config.epoch):
		
	last_time = time.time()
	total_loss = 0
	
	for x in range(0, len(label_list) - config.batch_size, config.batch_size):
		images = np.zeros((config.batch_size, config.image_size, config.image_size, 3))
		label = np.zeros((config.batch_size, config.no_grid, config.no_grid, config.no_boxes_per_cell + config.no_classes))
		
		for batch in range(config.batch_size):
			image_path = label_list[x + batch]['image_path']
			images[n, :, :, :] = utils.read_img(image_path)
			label[n, :, :, :] = label_list[x + n]['label']

		loss, _ = sess.run([total_loss_placeholder, optimizer], feed_dict = {image_placeholder: images, label_placeholder: label})
		total_loss += loss

		if ((x + 1) % config.checkpoint == 0):
			print ("checkpoint reached: {}".format(str(x + 1)))
	
	np.random.shuffle(label_list)
	current_status ="epoch: " + str(epoch_no + 1) +" , loss: " + str(loss / (len(label_list) - config.batch_size / (config.batch_size * 1.0))) " ,  time (s) / epoch: " + str(time.time() - last_time)
	print (current_status)

	
	with open(log_file, "a") as myfile:
    	myfile.write(current_status)
    	myfile.write("\n")

    myfile.close()

    
	saver.save(sess, checkpoint_path)