# """YOLO V3 TRAINING MODULE"""
# import os
import torch

import cfgs.config as cfg #make a common config file

# import datetime

# try:
#     from pycrayon import CrayonClient
# except ImportError:
#     CrayonClient = None

# # from datasets.pascal_voc import VOCDataset
# # import utils.yolo as yolo_utils
# # import utils.network as net_utils

from darknet import Darknet


from utils.timer import Timer

from random import randint



import torchvision
import argparse

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

from custom_dataset.bbdset import dataset as lmdb




##########FUNCTIONS#################
#Pending
##Loading from a checkpoint 
##parsing the target dictionaries


def arg_parse():
    """
    Parse arguements to the training module
    
    """
    
    
    parser = argparse.ArgumentParser(description='Training module')
   
    parser.add_argument("-i", dest = 'images', help = 
                        "path to train image directory",
                        default = "imgs", type = str)
    parser.add_argument("-w", dest = 'workers', help = 
                        "number of workers to load the images",
                        default = "4", type = int)
    parser.add_argument("-b", dest = "batch", help = "Batch size", default = 50, type = int)
    # parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    # parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("-c", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("-t", dest = "use_tensorboard", help = "Disable tensorboard", default = True, type = bool)
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

 	args = argparse()

    #Use LMDB or not
    if args.lmdb == 0:
        image_data = torchvision.datasets.ImageFolder(args.path)
        data_loader = torch.utils.data.DataLoader(image_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        #load the annotations

    else :
        #custom dataset pipeline with LMDB
        dataset = lmdb(target_file, root_dir, transforms)
        data_loader = torch.utils.data.DataLoader(image_data, batch_size = args.batch, shuffle=True, num_workers = args.workers)
        #currently contains dict with keys : 'image' and 'targets'



    
    net = Darknet(args.cfgfile)
    with open(args.cfgfile, r) as config:
        cfg = config

    

    #load from a checkpoint

    net.cuda()
    net.train()

    print('network loaded')

    #Optimizer
    start_epoch = 0
    lr = cfg.learning_rate
    Optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.decay)

    #tensorboard
    if args.use_tensorboard and SummaryWriter is not None:
        summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
    # else:
    #     summary_writer = None

    batch_per_epoch = dataloader.batch_per_epoch
    train_loss = 0
    bbox_loss, iou_loss, cls_loss = 0.,0.,0.
    cnt = 0
    size_index = 0
    t = Timer()
    for step in range(start_epoch * dataloader.batch_per_epoch, cfg.max_epoch * dataloader.batch_per_epoch):
        t.tic()

        #batch
        batch = dataloader.next_batch(size_index)
        im = batch['image']
        gt_boxes = batch['gt_classes']
        gt_classes = batch['gt_classes']
        # dontcare = batch['dontcare']
        # origin_im = batch['origin_im']

        #forward
        im_data = net_utils.np_to_variable(im, is_cuda = True, volatile=False).permute(0,3,1,2)
        bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)

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
               (imdb.epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                iou_loss, cls_loss, duration,
                str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))

            if summary_writer and step % cfg.log_interval == 0 :
                summary_writer.add_scalar('loss_train', train_loss, step)
                summary_writer.add_scalar('loss_bbox', bbox_loss, step)
                summary_writer.add_scalar('loss_iou', iou_loss, step)
                summary_writer.add_scalar('loss_cls', cls_loss, step)
                summary_writer.add_scalar('learning_rate', lr, step)

            # plot the results
                bbox_pred = bbox_pred.data[0:1].cpu().numpy()
                iou_pred = iou_pred.data[0:1].cpu().numpy()
                prob_pred = prob_pred.data[0:1].cpu().numpy()

                image = im[0]

                bboxes, scores, cls_inds = yolo_utils.postprocess(
                    bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh=0.3, size_index=size_index)
                im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
                summary_writer.add_image('predict', im2show, step)


            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0.,0.,0.

            cnt = 0
            t.clear()

            size_index = randint(0, len(cfg.multiscale_inp_size) -1)
            print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))


        if step > 0 and (step % dataloader.batch_per_epoch == 0):
            if imdb.epoch in cfg.lr_decay_epochs:
                lr *= cfg.lr_decay
                optimizer = torch.optime.SGD(net.parameters(), lr =lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


            save_name = os.path.join(cfg.train_output_dir,
                                     '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
            net_utils.save_net(save_name, net)
            print(('save model: {}'.format(save_name)))
            step_cnt = 0


    dataloader.close()