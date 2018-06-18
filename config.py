#Config file

dir_path = "/home/sagarwal311/practice_dl/object_detection_traffic_sign/data"
log_file_path = "/home/sagarwal311/practice_dl/object_detection_traffic_sign/log"
checkpoint_path = "/home/sagarwal311/practice_dl/object_detection_traffic_sign/checkpoint"
pretrained_model_path = "/home/sagarwal311/practice_dl/object_detection_traffic_sign/yolo.ckpt"
checkpoint = 200

image_size = 512
image_width = 2048 
image_height = 2048

dataset_ratio = 0.5
batch_size = 40
no_boxes_per_cell = 7
no_grid = 10
no_classes = 221

class_scale = 0.87
confidence_obj_scale = 0.9
confidence_noobj_scale = 0.4
coord_scale = 0.8

epoch = 100
learning_rate = 0.00025
alpha = 0.1
dropout = 0.2