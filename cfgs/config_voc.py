import numpy as np


# trained model
h5_fname = 'yolo-voc.weights.h5'

# VOC classes
# Replace the value of 'classes' tuple with your own custom classes
classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(classes)
class_map = {j: i for i, j in list(enumerate(classes))}

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                      (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
                     dtype=np.float)
num_anchors = len(anchors)
