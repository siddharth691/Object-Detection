#Config file

dir_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/data"
log_file_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/log"
checkpoint_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/checkpoint"
pretrained_model_path = "/home/siddharth/Desktop/Adversarial Learning SP/DL/object_detection/yolo.ckpt"
checkpoint = 200

image_size = 448
image_width = 2048 
image_height = 2048

dataset_ratio = 1
batch_size = 20
no_boxes_per_cell = 7
no_grid = 10
no_classes = 221

class_scale = 0.87
confidence_obj_scale = 0.9
confidence_noobj_scale = 0.4
coord_scale = 0.8

epoch = 300
learning_rate = 0.0001
alpha = 0.1
dropout = 0.2