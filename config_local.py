#Config file

dir_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/data"
log_file_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/log"
checkpoint_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/checkpoint"
pretrained_model_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/yolo.ckpt"
checkpoint = 200

image_size = 1024
image_width = 2048 
image_height = 2048

dataset_ratio = 0.01
batch_size = 2
no_boxes_per_cell = 7
no_grid = 10
no_classes = 221

class_scale = 2.0
confidence_obj_scale = 2.0
confidence_noobj_scale = 1.0
coord_scale = 5.0

epoch = 300
learning_rate = 0.0001
alpha = 0.1
dropout = 0.5