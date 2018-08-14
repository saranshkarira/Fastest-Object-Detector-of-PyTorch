# General modules
import os
import sys
import datetime
from random import randint
import argparse
import time

# DL specific modules
import torch
import torchvision
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

# Imports from within API
import cfgs.config as cfg  # make a common config file
import utils.network as net_utils  # THEY HAVE ALTERNATES
from darknet import Darknet19 as Darknet
from utils.timer import Timer
from dataset import dataset as dset
from loss import loss


# Parse the Arguments
def arg_parse():
    """
    Parse arguements to the training module
    -i 'path/to/image_dir'
    -w 'num of workers'
    -b 'batch size'
    -tl 'True/False' {activate transfer learning}
    -c 'preferred yolo flavor'
    -t 'True/False' {Use Tensorboard}

    """

    parser = argparse.ArgumentParser(description='Training module')

    parser.add_argument("-i", dest='images', help="path to train image directory",
                        default="imgs", type=str)
    parser.add_argument("-w", dest='workers', help="number of workers to load the images",
                        default="4", type=int)
    parser.add_argument("-b", dest="batch", help="Batch size", default=30, type=int)

    parser.add_argument("-tl", dest='transfer', help='transfer_learning', default=False, type=bool)

    parser.add_argument("-c", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("-t", dest="use_tensorboard", help="Disable tensorboard", default=True, type=bool)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    # Use LMDB custom dataset or VOC-style
    if cfg.lmdb:
        dataset = dset(cfg.target_file, cfg.root_dir, cfg.multi_scale_inp_size)  # , cfg.transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    else:

        image_data = torchvision.datasets.ImageFolder(args.path)
        data_loader = torch.utils.data.DataLoader(image_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, multiscale=cfg.multi_scale_inp_size)

    # replace 4 with the number of classes in your custom dataset
    classes = 20 if args.transfer else cfg.num_classes

    # create the network
    net = Darknet(classes)

    # Load weights

    # Loads pretrained yolo VOC weights
    if args.transfer:
        net.load_from_npz(cfg.pretrained_model, num_conv=18)
        exp_name = str(int(time.time()))  # For tensorboard consistency on reloads
        start_epoch = 0
        j = 0
        lr = cfg.init_learning_rate

    # Loads from a latest saved checkpoint in case training takes place over multiple days.
    else:
        path_t = cfg.trained_model()
        if os.path.exists(path_t):
            j, exp_name, start_epoch, lr = net_utils.load_net(path_t, net)
            j, exp_name, start_epoch, lr = int(j), str(int(exp_name)), int(start_epoch), float(lr)
            print('lr is {} and its type is {}'.format(lr, type(lr)))
        else:
            e = 'no checkpoint to load from\n'
            sys.exit(e)

    # To keep the tensorflow logs consistent in case training takes multiple days with a lot of start and stops
    path = os.path.join(cfg.TRAIN_DIR, 'runs', str(exp_name))
    if not os.path.exists(path):
        os.makedirs(path)

    # Transfer learning
    # Freeze all the parameters
    for params in net.parameters():
        params.requires_grad = False
    # Replace the last conv5 module
    if args.transfer:
        shape = net.conv5.conv.weight.shape
        new_layer = net_utils.Conv2d(shape[1], 45, shape[2], 1, relu=False)
        net.conv5 = new_layer  # make it generalizable
    # Unfreeze last module's params
    for params in net.conv5.parameters():
        params.requires_grad = True

        print('Tranfer Learning Active')

    # os.environ['CUDA_VISIBLE_DEVICES'] = 0, 1, 2
    # torch.cuda.manual_seed(seed)
    # net = torch.nn.DataParallel(net).cuda()

    net.train()
    print('network loaded')

    # Optimizer only optimizes 5th conv layer
    optimizable = net.conv5.parameters  # this is always the case whether transfer or not

    net.cuda()
    # net = torch.nn.DataParallel(net)
    # device = torch.device("cuda:0")
    # net.to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

    # Load the model on gpu
    # net.cuda()

    # SGD optimizer
    optimizer = torch.optim.SGD(optimizable(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # tensorboard
    if args.use_tensorboard and SummaryWriter is not None:
        summary_writer = SummaryWriter(path)

    batch_per_epoch = dataset.length / args.batch
    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0., 0., 0.
    cnt = 0
    step_cnt = 0
    size_index = 0
    t = Timer()
    epoch = start_epoch
    print('this')
    for step in range(int(epoch), cfg.max_epoch):

        # batch
        for i, batch_of_index in enumerate(dataloader):
            t.tic()

            # OG yoloV2 changes scales every 10 epochs
            # Selecting index first thing than last otherwise one scale gets more trained than others due to multiple start-stops

            if i % 10 == 0:
                size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
                print('new scale is {}'.format(cfg.multi_scale_inp_size[size_index]))

            batch = dataset.fetch_parse(batch_of_index, size_index)
            im = batch['images']
            gt_boxes = batch['gt_boxes']
            gt_classes = batch['gt_classes']
            dontcare = batch['dontcare']
            origin_im = ['origin_im']

            # sending images onto gpu after turning them into torch variable
            im = net_utils.np_to_variable(im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)

            bbox_pred, iou_pred, prob_pred = net(im)

            bbox_loss_i, iou_loss_i, cls_loss_i = loss(gt_boxes, gt_classes, dontcare, size_index, bbox_pred, iou_pred, prob_pred)

            # accumulating mini-batch loss
            loss = bbox_loss_i + iou_loss_i + cls_loss_i
            bbox_loss += bbox_loss_i.data.cpu().numpy()[0]
            iou_loss += iou_loss_i.data.cpu().numpy()[0]
            cls_loss += cls_loss_i.data.cpu().numpy()[0]
            train_loss += loss.data.cpu().numpy()[0]

            # clearing grads before calculating new ones and then updating wts
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            step_cnt += 1
            j += 1
            duration = t.toc()

            # print average loss of the mini-batch
            if cnt % cfg.disp_interval == 0:

                train_loss /= cnt
                bbox_loss /= cnt
                iou_loss /= cnt
                cls_loss /= cnt

                print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
                       'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
                       (step, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                        iou_loss, cls_loss, duration,
                        str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))

                # write tensorboard logs of the average loss
                summary_writer.add_scalar('loss_train', train_loss, j)
                summary_writer.add_scalar('loss_bbox', bbox_loss, j)
                summary_writer.add_scalar('loss_iou', iou_loss, j)
                summary_writer.add_scalar('loss_cls', cls_loss, j)
                summary_writer.add_scalar('learning_rate', lr, j)

                train_loss = 0
                bbox_loss, iou_loss, cls_loss = 0., 0., 0.
                cnt = 0
                t.clear()

        # learning rate decay every said epochs
        if step in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(optimizable(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        # write parameters and hyper-parameters to checkpoints
        train_output_dir = os.path.join(cfg.TRAIN_DIR, 'checkpoints', exp_name)
        cfg.mkdir(train_output_dir, max_depth=3)
        save_name = os.path.join(train_output_dir, '{}.h5'.format(step))
        net_utils.save_net(j, exp_name, step + 1, lr, save_name, net)
        print(('save model: {}'.format(save_name)))

        # clean old checkpoints
        if step % 10 == 1:
            cfg.clean_ckpts(train_output_dir)

        # counter reset
        step_cnt = 0
