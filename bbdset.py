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

# Dataclass


class dataset(data.Dataset):
    def __init__(self, target_file, root_dir, transforms=True):
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
        self.env = lmdb.open(root_dir, max_readers=5, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        with self.txn.cursor() as cursor:
            self.length = self.txn.stat()['entries'] - 1  # for mapping
            mapping = cursor.get('mapping')
            self.mapping = pickle.loads(mapping)  # mapping.decode('base64', 'strict'))

        self.class_map = {'Weapon': [1, 0, 0, 0], 'Vehicle': [0, 1, 0, 0], 'Building': [0, 0, 1, 0], 'Person': [0, 0, 0, 1]}

        with open(target_file) as opener:
            self.targets = json.load(opener)

# len

    def __len__(self):

        return self.length
    # getitem

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir, self.target_file.iloc[idx,0])
        # image = io.imread(img_name)

        # key = 'image' + str(idx)
        print(idx)
        image_id = self.mapping[idx]
        with self.txn.cursor() as cursor:
            data = cursor.get(image_id)  # rebuild dataloader library to load more than single index a time
        # buf = six.BytesIO()
        # buf.write(data)
        # buf.seek(0)
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

        # targets = self.target_file.iloc[idx,1:].as_matrix()
        # gt_boxes = gt_boxes.astype('float').reshape(-1, 2)  # add the reshape dims

        sample = {'image': image, 'gt_classes': np.asarray(gt_classes), 'gt_boxes': np.asarray(gt_boxes)[0].reshape(-1, 2)}
        # print(sample['gt_boxes'])
        if self.transform or True:
            # print('dog_2')
            rescale = Rescale(500)
            sample = rescale(sample)
            random_crop = RandomCrop(416)
            sample = random_crop(sample)
            # to_tensor = ToTensor() ## takes numpy input
            # sample = to_tensor(sample)

        # print('here')
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

        img = sktransform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        gt_boxes = gt_boxes * [new_w / w, new_h / h]

        return {'image': img, 'gt_boxes': gt_boxes, 'gt_classes': sample['gt_classes']}

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

        return {'image': image, 'gt_boxes': gt_boxes, 'gt_classes': sample['gt_classes']}

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
