import cv2
import lmdb
import json
import random
import os
import glob
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db"))
root_dir = os.path.join(path, "image_data")
targets_dir = os.path.join(path, "targets")

class_map = {'Weapon': [0], 'Vehicle': [1], 'Building': [2], 'People': [3]}

env = lmdb.open(root_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

if os.path.exists(glob.glob(os.path.join(targets_dir, '*.json'))[0]):
    target_file = glob.glob(os.path.join(targets_dir, '*.json'))[0]
    with open(target_file) as opener:
        targets = json.load(opener)

# image_ids =


recs = {}


for i in targets:
    gt_boxes = []
    gt_classes = []
    imagenames = []
    for k, v in i.iteritems():
        if k == 'Var1':
            img = v.split('/')[-1]

        if len(v) == 0:
            pass

        elif isinstance(v[0], (int)):
            gt_boxes.append(v)
            gt_classes.append(class_map[k])

        elif isinstance(v[0], (list)):
            for anns in v:
                gt_boxes.append(anns)
                gt_classes.append(class_map[k])
    imagenames.append(img)
    recs[img] = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes}

    imagenames = random.shuffle(imagenames)
    # since imagenames are shuffled access everything using this list as index

    with env.begin(write=False) as txn:
        with txn.cursor() as cursor:
            for i in range(len(imagenames)):
                images = cursor.get(imagenames[i].encode())


def scale_crop(images, scale):
    for i in range(len(images)):

        # new_shape = []
        im = images[i]
        h, w = im.shape[:2]
        if h > w:
            new_h, new_w = scale * h / w, scale
            cv2.resize(im, (new_h.new_w))
            top = np.randint(0, h - new_h)
            left = 0
        else:
            new_h, new_w = scale, scale * w / h
            top = 0
            left = np.randint(0, w - new_w)
        im = cv2.resize(im, (new_h.new_w))
        im = im[top: top + new_h, left: left + new_w]
        images[i] = im

    return images


def grid_stitch(scale, grid, p1, p2):
    """ Grid Stitch function,
    Scale is the scale you want your resultant image to be in.
    p1 and p2 are boundary percents
    Grid is the number of images we want to stich.
    Scale 448  = 448x448
    p1, p2 = 25, 50 == Object sizes are between 25 and 50% of the image
    grid 2 = 2x2
     """

    box_area1 = ((scale**2) * (p1**2)) / 100

    box_area2 = ((scale**2) * (p2**2)) / 100
    s_images = []
    s_names = []
    for i in range(len(imagenames)):
        for gt_box in recs[imagenames[i]]['gt_boxes']:
            if gt_box > box_area1 and gt_box < box_area2:
                s_images.append(images[i])
                s_names.append(imagenames[i])
                break
    counter = 0
    # s_images = random.shuffle(s_images)
    # after all images are on the same scale (448x448)
    s_images = scale_crop(s_images, grid)

    for i in range(0, len(s_images), grid**2):
        imgs = np.asarray(s_images[i:i + grid**2]).reshape(grid, grid, scale, scale, 3)
        hstacks = [np.hstack(imgs[i, :, :, :, :]) for i in range(grid)]
        grid = np.vstack(hstacks)
        cv2.imwrite(str(counter) + '.jpg', grid)
        counter += 1
