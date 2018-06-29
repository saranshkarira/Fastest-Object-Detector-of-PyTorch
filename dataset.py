# DataSet Barebone

import torch.utils.data as data

import numpy as np

import cv2

import json
import lmdb

# from utils.im_transform put your transforms here
import threading

import sys
from cfgs import config as cfg
import time
import os
from eval_voc import voc_eval
import pickle


class dataset(data.Dataset):
    def __init__(self, target_file, root_dir, multiscale, train=True):  # , transforms=True):
        """
        Args:
            target_file (string): Path to the target file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.crop = 320
        self.root_dir = root_dir
        self.train = train
        self.eval_name = str(time.time()) + '{}.txt'
        self.year = 2007  # eval metric

        self.env = lmdb.open(root_dir, max_readers=5, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.length = self.txn.stat()['entries'] - 4  # for mapping

        classes = None
        if classes is None:
            self.classes = {'Weapon', 'Vehicle', 'Building', 'People'}
        else:
            self.classes = classes

        self.class_map = {'Weapon': [0], 'Vehicle': [1], 'Building': [2], 'People': [3]}

        with open(target_file) as opener:
            self.targets = json.load(opener)

        self.dst_size = multiscale
        # self.sample = {'image': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': []}
    # len

    def __len__(self):

        return self.length
    # getitem

    def __getitem__(self, idx):
        return idx

    # ### START #####

    def imcv2_recolor(self, im, a=.1):

        t = np.random.uniform(-1, 1, 3)

        # random amplify each channel
        im = im.astype(np.float)
        im *= (1 + t * a)  # (* 1.0x, x is a random value)
        mx = 255. * (1 + a)
        up = np.random.uniform(-1, 1)
        im = np.power(im / mx, 1. + up * .5)
        # return np.array(im * 255., np.uint8)
        return im

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

    # def multiscale(self, inp_size, gt_boxes, im):

    #     if inp_size is not None:
    #         w, h = inp_size

    #         try:
    #             gt_boxes[:, 0::2] *= float(w) / im.shape[1]
    #             gt_boxes[:, 1::2] *= float(h) / im.shape[0]
    #         except IndexError:
    #             print(gt_boxes.shape)
    #             sys.exit(1)
    #         im = cv2.resize(im, (h, w))
    #     print(im.shape)
    #     return gt_boxes, im

    def multiscale(self, inp_size, gt_boxes, im):

        if inp_size is not None:
            _, scale = inp_size
            h, w = im.shape[:2]

            if h > w:
                new_h, new_w = scale * h / w, scale

            elif w > h:
                new_h, new_w = scale, scale * w / h
            else:
                new_h, new_w = scale, scale

            try:
                gt_boxes[:, 0::2] *= int(float(new_w) / im.shape[1])
                gt_boxes[:, 1::2] *= int(float(new_h) / im.shape[0])
            except IndexError:
                print(gt_boxes.shape)
                sys.exit(1)
            im = cv2.resize(im, (new_h, new_w))
            # print(im.shape)
        return gt_boxes, im

    def flip(self, im, boxes):
        if len(boxes) == 0:
            return boxes

        boxes = boxes

        flip = np.random.uniform() > 0.5
        if flip:
            im = cv2.flip(im, 1)
            boxes_x = np.copy(boxes[:, 0])
            boxes[:, 0] = im.shape[1] - boxes[:, 2]
            boxes[:, 2] = im.shape[1] - boxes_x

        return im, boxes

    def random_crop(self, output_size, im, gt_boxes):
        """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2

        h, w = im.shape[:2]
        new_h, new_w = output_size
        if h != new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if w != new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        im = im[top: top + new_h, left: left + new_w]
        # print(im)

        gt_boxes = gt_boxes - [left, top, left, top]
        # print(im.shape)

        return im, gt_boxes

    def get_annots(self, index):
        gt_boxes = []
        gt_classes = []
        # image_id = self.targets[index]['Var1'].split('/')[-1].encode()
        for k, v in self.targets[index].iteritems():

            if k == 'Var1':
                image_id = v.split('/')[-1]

            elif len(v) == 0:
                pass

            elif isinstance(v[0], (int)):
                gt_boxes.append(v)
                gt_classes.append(self.class_map[k])

            elif isinstance(v[0], (list)):
                for anns in v:
                    gt_boxes.append(anns)
                    gt_classes.append(self.class_map[k])

        return image_id, gt_boxes, gt_classes

    def preprocess_train(self, index, size_index, multi_scale_inp_size):

        inp_size = multi_scale_inp_size[size_index]

        image_id, gt_boxes, gt_classes = self.get_annots(index)

        # use map function here
        with self.txn.cursor() as cursor:  # cannot append before preprocess
            im = cursor.get(image_id.encode())  # check cursor for list parsing #append because we are getting images manually now

        im = cv2.imdecode(np.fromstring(im, dtype=np.uint8), 1)

        # transforms:

        ori_im = np.copy(im)

        gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
        gt_boxes, im = self.multiscale(inp_size, gt_boxes, im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Don't run tfms below during eval/test
        if self.train:
            if self.crop:
                im, gt_boxes = self.random_crop(self.crop, im, gt_boxes)

            im, gt_boxes = self.flip(im, gt_boxes)

            im = self.imcv2_recolor(im)

            gt_boxes = self.clip_boxes(gt_boxes, im.shape)

        return im, gt_boxes, gt_classes, [], ori_im

    def fetch_batch(self, ith, index, size_index, dst_size):
        images, gt_boxes, classes, dontcare, origin_im = self.preprocess_train(index, size_index, dst_size)
        # print(ith)
        self.batch['images'][ith] = images
        self.batch['gt_boxes'][ith] = gt_boxes
        self.batch['gt_classes'][ith] = classes
        self.batch['dontcare'][ith] = dontcare
        self.batch['origin_im'][ith] = origin_im
        return 0

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
        # print(self.batch['images'])
        self.batch['images'] = np.asarray(self.batch['images'], dtype=np.float64)

        # self.batch = self.to_tensor()
        return self.batch

    # ######END#######

    # ######START EVAL#######

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        # if self.config['cleanup']:
        #     for cls in self._classes:
        #         if cls == '__background__':
        #             continue
        #         filename = self._get_voc_results_file_template().format(cls)
        #         os.remove(filename)

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self.eval_name.format(cls)
            with open(filename, 'wt') as f:
                for index in range(self.length):
                    dets = all_boxes[cls_ind][index]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    image_id = self.targets[index]['Var1'].split('/')[-1]
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(image_id, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = cfg.target_file
        cachedir = os.path.join(cfg.TEST_DIR, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.eval_name.format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
