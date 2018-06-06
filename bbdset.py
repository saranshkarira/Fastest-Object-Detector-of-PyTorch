# DataSet Barebone
# sT# targets are treated as a [-1,2] matrix of points of polygon with 2 coordinates
# Imports
import torch.utils.data as data
import torch
# import pandas as pd  # what about hdf5
# import os
import numpy as np
# import six
# from PIL import Image
import cv2
from skimage import transform as sktransform
import json
import lmdb
from datasets.imdb import ImageDataset
import pickle
from utils.im_transform import imcv2_affine_trans, imcv2_recolor
import threading
# import torchvision.transforms
# import pprint
# from PIL import Image
# Dataclass


class dataset(data.Dataset, ImageDataset):
    def __init__(self, target_file, root_dir, multiscale, transforms=True):
        """
        Args:
            target_file (string): Path to the target file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.target_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.env = lmdb.open(root_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        with self.txn.cursor() as cursor:
            self.length = self.txn.stat()['entries'] - 1  # for mapping
            mapping = cursor.get('mapping')
            self.mapping = pickle.loads(mapping)  # mapping.decode('base64', 'strict'))
        classes = None
        if classes is None:
            self._classes = {'Weapon', 'Vehicle', 'Building', 'Person'}
        else:
            self._classes = classes

        self.class_map = {'Weapon': [0], 'Vehicle': [1], 'Building': [2], 'Person': [3]}

        with open(target_file) as opener:
            self.targets = json.load(opener)

        self._salt = str()
        self.dst_size = multiscale
        self.sample = {'image': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': []}
# len

    def __len__(self):

        return self.length
    # getitem

    def __getitem__(self, idx):
        return idx

    # ### START #####

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def _offset_boxes(self, boxes, im_shape, scale, offs, flip):
        if len(boxes) == 0:
            return boxes
        boxes = np.asarray(boxes, dtype=np.float)
        boxes *= scale
        boxes[:, 0::2] -= offs[0]
        boxes[:, 1::2] -= offs[1]
        boxes = self.clip_boxes(boxes, im_shape)

        if flip:
            boxes_x = np.copy(boxes[:, 0])
            boxes[:, 0] = im_shape[1] - boxes[:, 2]
            boxes[:, 2] = im_shape[1] - boxes_x

        return boxes

    def preprocess_train(self, index, size_index, multi_scale_inp_size):
        # data = []
        # images = []
        inp_size = multi_scale_inp_size[size_index]

        image_id = self.mapping[index]  # use map function here
        with self.txn.cursor() as cursor:  # cannot append before preprocessZ
            im = cursor.get(image_id)  # check cursor for list parsing #append becouse we are getting images manually now

        im = cv2.imdecode(np.fromstring(im, dtype=np.uint8), 1)  # make it Pool process

        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) already applied in affine tfms
        # send to prepackaging tho
        # Targets

        gt_boxes = []
        gt_classes = []

        # keys = self.targets[idx].keys()

        # do it above image as this is lower cost op, this will reduce lock convoy and make it concurrent if possible
        for k, v in self.targets[index].iteritems():
            if len(v) == 0:
                continue

            elif isinstance(v[0], (int)):
                gt_boxes.append(v)
                gt_classes.append(self.class_map[k])

            elif isinstance(v[0], (list)):
                for anns in v:
                    gt_boxes.append(anns)
                    gt_classes.append(self.class_map[k])

        ori_im = np.copy(im)
        # print(im.shape)
        im, trans_param = imcv2_affine_trans(im)

        # analyse the below twos
        scale, offs, flip = trans_param
        gt_boxes = self._offset_boxes(gt_boxes, im.shape, scale, offs, flip)

        if inp_size is not None:  # bbox axis is flipped from the image axis, or is it??
            w, h = inp_size
            gt_boxes[:, 0::2] *= float(w) / im.shape[1]
            gt_boxes[:, 1::2] *= float(h) / im.shape[0]
            im = cv2.resize(im, (w, h))

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = imcv2_recolor(im)
        # im /= 255.

        # im = imcv2_recolor(im)
        # h, w = inp_size
        # im = cv2.resize(im, (w, h))
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im /= 255
        # gt_boxes = np.asarray(boxes, dtype=np.int)
        return im, gt_boxes, gt_classes, [], ori_im

    def fetch_batch(self, ith, index, size_index, dst_size):
        images, gt_boxes, classes, dontcare, origin_im = self.preprocess_train(index, size_index, dst_size)
        print(ith)
        self.batch['images'][ith] = images
        self.batch['gt_boxes'][ith] = gt_boxes
        self.batch['gt_classes'][ith] = classes
        self.batch['dontcare'][ith] = dontcare
        self.batch['origin_im'][ith] = origin_im

    # def to_tensor(self):
    #     sample = self.batch
    #     # swap color axis because
    #     # numpy image: H x W x C
    #     # torch image: C X H X W
    #     # image = image.transpose((2, 0, 1))
    #     return {'images': torch.Tensor(sample['images']),
    #             'gt_boxes': torch.from_numpy(sample['gt_boxes']),
    #             'gt_classes': torch.Tensor(sample['gt_classes']),
    #             'dontcare': torch.from_numpy(np.asarray(sample['dontcare']))}

    def fetch_parse(self, index, size_index):
        index = index.numpy()
        lenindex = len(index)
        self.batch = {'images': [list()] * lenindex,
                      'gt_boxes': [list()] * lenindex,
                      'gt_classes': [list()] * lenindex,
                      'dontcare': [list()] * lenindex,
                      'origin_im': [list()] * lenindex}

        ths = []

        for ith in range(lenindex):
            ths.append(threading.Thread(target=self.fetch_batch, args=(ith, index[ith], size_index, self.dst_size)))
            ths[ith].start()
        for ith in range(lenindex):
            ths[ith].join()
        # x = [(x.shape) for x in self.batch['gt_boxes']]
        # print(np.asarray(self.batch['gt_boxes']).shape, 'goda')
        self.batch['images'] = np.asarray(self.batch['images'])
        # self.batch['gt_boxes'] = self.batch['gt_boxes']
        # self.batch = self.to_tensor()
        return self.batch
        # images, gt_boxes, classes, dontcare, origin_im = preprocess_train(index, size_index, self.dst_size)

    # ######END#######

    # transforms

# Rescale


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt_boxes']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size

            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        sample['image'] = sktransform.resize(image, (new_h, new_w), mode='constant')

        sample['gt_boxes'] = gt_boxes * [new_w / w, new_h / h, new_w / w, new_h / h]


# RandomCrop


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt_boxes']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # print(new_h, new_w, h, w)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample['image'] = image[top: top + new_h, left: left + new_w]

        sample['gt_boxes'] = (
            gt_boxes - [left, top, left, top])
