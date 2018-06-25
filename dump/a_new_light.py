
# 1
for layer in model.modules():
    if isinstance(layer, nn.Conv2d):
        print("Pruning Conv-{}".format(index))
        filters = layer.weight.data.numpy()
        new_filters = prune_layer(filters)  # reshape the tensor

        # Update layer with new tensor
        layer.weight.data = torch.from_numpy(new_filters).float()
        layer.out_channels = new_filters.shape[0]

# 2
def __getitem__(self, idx):
    return idx

# 3
def fetch_batch_data(self, ith, x, size_index):
    images, gt_boxes, classes, dontcare = self._im_processor(
        [self.image_names[x], self.get_annotation(x), self.dst_size], None)

    # multi-scale
    # w, h = cfg.multi_scale_inp_size[size_index]
    # gt_boxes = np.asarray(gt_boxes, dtype=np.float)
    # if len(gt_boxes) > 0:
    #   gt_boxes[:, 0::2] += float(w)/ image.shape[1]
    #   gt_boxes[:, 1::2] += float(h)/ image.shape[0]
    # images = cv2.resize(images, (w,h))

    self.batch['images'][ith] = images
    self.batch['gt_boxes'][ith] = gt_boxes
    self.batch['gt_classes'][ith] = classes
    self.batch['dontcare'][ith] = dontcare
    # self.batch[]

# 4
def parse(self, index, size_index):
    index = index.numpy()
    lenindex = len(index)
    self.batch = {'images': [] + list(lenindex),
                  'gt_boxes': [] + list(lenindex),
                  'gt_classes': [] + list(lenindex),
                  'dontcare': [] + list(lenindex),
                  'origin_im': [] + list(lenindex)

                  }
    ths = []  # parsing fxn
    for ith in range(lenindex):

    return self.batch


# 5

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

    self.sample = {'image': image, 'gt_classes': np.asarray(gt_classes), 'gt_boxes': np.asarray(gt_boxes).reshape(-1, 4), 'dontcare': np.asarray(self.multiscale)}  # .reshape(-1, 2)}

    if self.transform or True:
        rescale = Rescale(500)
        rescale(self.sample)
        random_crop = RandomCrop(416)
        random_crop(self.sample)
        # totensor = ToTensor()
        # totensor(self.sample)
        # print(sample['image'].shape)

    self.sample['image'] = np.rollaxis(self.sample['image'], axis=2, start=0)
    # sample = [sample]

    # sample['image'] = Image.fromarray(sample['image'])
    # self.tubelight['image'].append(sample['image'])
    # self.tubelight['gt_classes'].append(sample['gt_classes'])
    # self.tubelight['gt_boxes'].append(sample['gt_boxes'])
    # self.tubelight['dontcare'].append(sample['dontcare'])

    # self.tubelight['image'] = np.asarray(self.tubelight['image'])
    print([self.sample[key].shape for key in self.sample], '\n')
    return self.sample
