import os
import cv2
import numpy as np
import pickle
import argparse

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
# from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg
from dataset import dataset as dset


# Unused anyway
def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    # use this instead of preprocess train but it does not rescale gTruths
    im_data = np.expand_dims(yolo_utils.preprocess_test(image, cfg.multi_scale_inp_size), 0)  # noqa
    return image, im_data


parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--image_size_index', type=int, default=0,
                    metavar='image_size_index',
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
args = parser.parse_args()


# hyper-parameters
# ------------


output_dir = os.path.join(cfg.DATA_DIR, 'detections')

max_per_image = 300
thresh = 0.01
vis = False
# ------------


def test_net(net, dataloader, max_per_image=300, thresh=0.5, vis=False, dataset):
    x, _ = enumerate(dataloader)
    num_images = len(x)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(dataset.classes))]  # classes is in dataset actually

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index

    for index, batch_index in enumerate(dataloader):

        batch = dataset.fetch_parse(batch_index, size_index=size_index)  # How to pass size index
        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=True,
                                           volatile=True).permute(0, 3, 1, 2)

        _t['im_detect'].tic()
        bbox_pred, iou_pred, prob_pred = net(im_data)  # compare for arguments

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,
                                                          iou_pred,
                                                          prob_pred,
                                                          ori_im.shape,
                                                          cfg,
                                                          thresh,
                                                          size_index
                                                          )
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()

        for j in range(len(dataset.classes)):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(len(dataset.classes))])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, len(dataset.classes)):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))  # noqa
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis:
            im2show = yolo_utils.draw_detection(ori_im,
                                                bboxes,
                                                scores,
                                                cls_inds,
                                                cfg,
                                                thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show,
                                     (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            cv2.imshow('test', im2show)
            cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    dataset.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # data loader
    dataset = dset(cfg.eval_target_file, cfg.eval_root_dir, cfg.multi_scale_inp_size, train=False)  # , cfg.transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    net = Darknet19()
    net_utils.load_net(cfg.trained_model(), net)

    net.cuda()
    net.eval()

    test_net(net, dataloader, max_per_image, thresh, vis, dataset)
