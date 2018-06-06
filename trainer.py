# """YOLO V3 TRAINING MODULE"""
# import os
import torch

import cfgs.config as cfg  # make a common config file
import os
# import numpy as np
# import datetime

# try:
#     from pycrayon import CrayonClient
# except ImportError:
#     CrayonClient = None

# # from datasets.pascal_voc import VOCDataset

# import utils.yolo as yolo_utils
import utils.network as net_utils  # THEY HAVE ALTERNATES

import datetime
from darknet import Darknet19 as Darknet
from utils.timer import Timer
from random import randint


import torchvision
import argparse

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

from bbdset import dataset as dset
# from torch.autograd import Variable

# #########FUNCTIONS#################
# Pending
# Loading from a checkpoint
# parsing the target dictionaries


# def torch_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
#     v = torch.autograd.Variable(x.type(dtype), volatile=volatile)
#     if is_cuda:
#         v = v.cuda()
#     return v


def arg_parse():
    """
    Parse arguements to the training module

    """

    parser = argparse.ArgumentParser(description='Training module')

    parser.add_argument("-i", dest='images', help="path to train image directory",
                        default="imgs", type=str)
    parser.add_argument("-w", dest='workers', help="number of workers to load the images",
                        default="4", type=int)
    parser.add_argument("-b", dest="batch", help="Batch size", default=50, type=int)
    # parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    # parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("-c", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("-t", dest="use_tensorboard", help="Disable tensorboard", default=True, type=bool)
    # parser.add_argument("--weights", dest = 'weightsfile', help =
    #                     "weightsfile",
    #                     default = "yolov3.weights", type = str)
    # parser.add_argument("--reso", dest = 'reso', help =
    #                     "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
    #                     default = "320", type = str)
    # parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
    #                     default = "1,2,3", type = str)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    lmdb = 1
    # Use LMDB or not
    if lmdb == 0:
        image_data = torchvision.datasets.ImageFolder(args.path)
        data_loader = torch.utils.data.DataLoader(image_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, multiscale=cfg.multi_scale_inp_size)
        # load the annotations

    else:
        # custom dataset pipeline with LMDB
        dataset = dset(cfg.target_file, cfg.root_dir, cfg.multi_scale_inp_size, cfg.transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        # currently contains dict with keys : 'image' and 'targets'

    net = Darknet()
    # with open(args.cfgfile, 'r') as config:
    #     cfg = config

    # load from a checkpoint
    if False:
        net.load_from_npz(cfg.pretrained_model, num_conv=18)  # doubts
    net.cuda()
    net.train()

    print('network loaded')

    # Optimizer
    start_epoch = 0
    lr = cfg.init_learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # tensorboard
    if args.use_tensorboard and SummaryWriter is not None:
        summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
    # else:
    #     summary_writer = None

    batch_per_epoch = dataset.length / args.batch
    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    cnt = 0
    step_cnt = 0
    size_index = 0
    t = Timer()
    for step in range(start_epoch, cfg.max_epoch):
        t.tic()
        # batman = [v for k, v in enumerate(dataloader)]

        # batch
        for i, batch_of_index in enumerate(dataloader):
            # batch = iter(dataloader).next()
            # batch = batch[batch_index]
            # im = [i[0]['image'] for i in batch]
            # gt_boxes = [i[0]['gt_boxes'] for i in batch]
            # gt_classes = [i[0]['gt_classes'] for i in batch]
            # dontcare = [i[0]['dontcare'] for i in batch]
            batch = dataset.fetch_parse(batch_of_index, size_index)
            im = batch['images']
            gt_boxes = batch['gt_boxes']
            gt_classes = batch['gt_classes']
            dontcare = batch['dontcare']
            origin_im = ['origin_im']
            print(cnt, 'I am fucking working')
            # forward
            try:
                im = net_utils.np_to_variable(im,
                                              is_cuda=True,
                                              volatile=False).permute(0, 3, 1, 2)
            except:
                continue

            bbox_pred, iou_pred, prob_pred = net(im.cuda(), gt_boxes=gt_boxes, gt_classes=gt_classes, dontcare=dontcare, size_index=size_index)

            # backward
            loss = net.loss
            bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
            iou_loss += net.iou_loss.data.cpu().numpy()[0]
            cls_loss += net.cls_loss.data.cpu().numpy()[0]

            train_loss += loss.data.cpu().numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            step_cnt += 1
            duration = t.toc()

        if step % cfg.disp_interval == 0:
            train_loss /= cnt
            bbox_loss /= cnt
            iou_loss /= cnt
            cls_loss /= cnt

            print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                   'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                   (dataloader.epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                    iou_loss, cls_loss, duration,
                    str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))

        if summary_writer and step % cfg.log_interval == 0:
            summary_writer.add_scalar('loss_train', train_loss, step)
            summary_writer.add_scalar('loss_bbox', bbox_loss, step)
            summary_writer.add_scalar('loss_iou', iou_loss, step)
            summary_writer.add_scalar('loss_cls', cls_loss, step)
            summary_writer.add_scalar('learning_rate', lr, step)

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            cnt = 0
            t.clear()
            size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)

        if step > 0 and (step % dataloader.batch_per_epoch == 0):
            if dataloader.epoch in cfg.lr_decay_epochs:
                lr *= cfg.lr_decay
                optimizer = torch.optime.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

            save_name = os.path.join(cfg.train_output_dir,
                                     '{}_{}.h5'.format(cfg.exp_name, dataloader.epoch))
            net_utils.save_net(save_name, net)
            print(('save model: {}'.format(save_name)))
            step_cnt = 0

    dataloader.close()
