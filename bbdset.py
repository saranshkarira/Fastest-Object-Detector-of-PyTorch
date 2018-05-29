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
import pickle
# import torchvision.transforms
# import pprint
# from PIL import Image
# Dataclass


class dataset(data.Dataset):
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

        self.class_map = {'Weapon': [0], 'Vehicle': [1], 'Building': [2], 'Person': [3]}

        with open(target_file) as opener:
            self.targets = json.load(opener)
        self.multiscale = multiscale
        self.tubelight = {'image': [], 'gt_boxes': [], 'gt_classes': [], 'dontcare': []}
# len

    def __len__(self):

        return self.length
    # getitem

    def __getitem__(self, idx):

        print(idx)
        image_id = self.mapping[idx]
        with self.txn.cursor() as cursor:
            data = cursor.get(image_id)  # rebuild dataloader library to load more than single index a time

        img = cv2.imdecode(np.fromstring(data, dtype=np.uint8), 1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # getting is a very often command so not making a function to prevent redirection overhead

        # Targets

        gt_boxes = []
        gt_classes = []

        # keys = self.targets[idx].keys()

        for k, v in self.targets[idx].iteritems():
            if len(v) == 0:
                continue

            elif isinstance(v[0], (int)):
                gt_boxes.append(v)
                gt_classes.append(self.class_map[k])

            elif isinstance(v[0], (list)):
                for anns in v:
                    gt_boxes.append(anns)
                    gt_classes.append(self.class_map[k])

        sample = {'image': image, 'gt_classes': np.asarray(gt_classes), 'gt_boxes': np.asarray(gt_boxes).reshape(-1, 4), 'dontcare': self.multiscale}  # .reshape(-1, 2)}

        if self.transform or True:
            rescale = Rescale(500)
            rescale(sample)
            random_crop = RandomCrop(416)
            random_crop(sample)
            print(sample['image'].shape)

        sample['image'] = np.rollaxis(sample['image'], axis=2, start=0)
        # sample = [sample]

        # sample['image'] = Image.fromarray(sample['image'])
        self.tubelight['image'].append(sample['image'])
        self.tubelight['gt_classes'].append(sample['gt_classes'])
        self.tubelight['gt_boxes'].append(sample['gt_boxes'])
        self.tubelight['dontcare'].append(sample['dontcare'])
        return self.tubelight

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

        sample['gt_boxes'] = gt_boxes - [left, top, left, top]

# ToTensor


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt_boxes = sample['image'], sample['gt_boxes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt_boxes': torch.from_numpy(gt_boxes), 'gt_classes': torch.from_numpy(sample['gt_classes'])}
