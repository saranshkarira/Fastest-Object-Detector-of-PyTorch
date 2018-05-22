# DataSet Barebone
# sT# targets are treated as a [-1,2] matrix of points of polygon with 2 coordinates
# Imports
import torch.utils.data as data
import torch
# import pandas as pd  # what about hdf5
# import os
import numpy as np
import six
from PIL import Image
from skimage import transform
import json
import lmdb
# import pprint

# Dataclass


class dataset(data.Dataset):
    def __init__(self, target_file, root_dir, transforms=None):
        """
        Args:
            target_file (string): Path to the target file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # self.target_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.env = lmdb.open(root_dir, max_readers=5, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

        self.class_map = {'weapon': 0, 'vehicle': 1, 'building': 2, 'person': 3}

        with open(target_file) as opener:
            self.targets = json.load(opener)

# len


def __len__(self):
    # with self.txn.cursor() as cursor:
    # length = txn.stat()['entries']1
    return self.txn.stat()['entries']
# getitem


def __getitem__(self, idx):
    # img_name = os.path.join(self.root_dir, self.target_file.iloc[idx,0])
    # image = io.imread(img_name)

    with self.txn.cursor() as cursor:
        data = cursor[idx]  # rebuild dataloader library to load more than single index a time
    buf = six.BytesIO()
    buf.write(data)
    buf.seek(0)
    image = Image.open(buf).Convert('RGB')

    # getting is a very often command so not making a function to prevent redirection overhead

    # Targets

    gt_boxes = []
    gt_classes = []

    # keys = self.targets[idx].keys()

    for k, v in self.targets[idx]:
        if len(v) == 0:
            continue

        elif len(v[0]) == 1:
            gt_boxes.append(v)
            gt_classes.append(k)

        else:
            for anns in v:
                gt_boxes.append(anns)
                gt_classes.append(k)

    # targets = self.target_file.iloc[idx,1:].as_matrix()
    gt_boxes = gt_boxes.astype('float').reshape(-1, 2)  # add the reshape dims

    sample = {'image': image, 'gt_classes': gt_classes, 'gt_boxes': gt_boxes}

    if self.transform:
        sample = self.transform(sample)

    return sample

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

        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        gt_boxes = gt_boxes * [new_w / w, new_h / h]

        return {'image': img, 'gt_boxes': gt_boxes}

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

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        gt_boxes = gt_boxes - [left, top]

        return {'image': image, 'gt_boxes': gt_boxes}

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
                'gt_boxes': torch.from_numpy(gt_boxes)}
