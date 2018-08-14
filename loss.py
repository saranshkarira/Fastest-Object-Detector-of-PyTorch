import utils.network as net_utils
from utils.cython_bbox import bbox_ious, anchor_intersections
from utils.cython_yolo import yolo_to_bbox
from functools import partial
import cfgs.config as cfg
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn


def loss_fxn(gt_boxes, gt_classes, dontcare, size_index, bbox_pred, iou_pred, prob_pred):

    bbox_pred_np = bbox_pred.data.cpu().numpy()
    iou_pred_np = iou_pred.data.cpu().numpy()
    # print('1')
    _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask = build_target(bbox_pred_np,
                                                                              gt_boxes,
                                                                              gt_classes,
                                                                              dontcare,
                                                                              iou_pred_np,
                                                                              size_index)
    # print('2')
    _boxes = net_utils.np_to_variable(_boxes)
    _ious = net_utils.np_to_variable(_ious)
    _classes = net_utils.np_to_variable(_classes)
    box_mask = net_utils.np_to_variable(_box_mask,
                                        dtype=torch.FloatTensor)
    iou_mask = net_utils.np_to_variable(_iou_mask,
                                        dtype=torch.FloatTensor)
    class_mask = net_utils.np_to_variable(_class_mask,
                                          dtype=torch.FloatTensor)

    num_boxes = sum((len(boxes) for boxes in gt_boxes))
    # print(num_boxes, 'here are the number of boxes')
    # _boxes[:, :, :, 2:4] = torch.log(_boxes[:, :, :, 2:4])
    box_mask = box_mask.expand_as(_boxes)

    bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes  # noqa
    iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes  # noqa

    class_mask = class_mask.expand_as(prob_pred)
    cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes  # noqa

    return bbox_loss, iou_loss, cls_loss


def build_target(bbox_pred_np, gt_boxes, gt_classes, dontcare,
                 iou_pred_np, size_index):
    """
    :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) :
                      (sig(tx), sig(ty), exp(tw), exp(th))
    """
    pool = Pool(processes=10)
    bsize = bbox_pred_np.shape[0]

    targets = pool.map(partial(process_batch, size_index=0),
                       ((bbox_pred_np[b], gt_boxes[b],
                         gt_classes[b], dontcare[b], iou_pred_np[b])
                        for b in range(bsize)))

    _boxes = np.stack(tuple((row[0] for row in targets)))
    _ious = np.stack(tuple((row[1] for row in targets)))
    _classes = np.stack(tuple((row[2] for row in targets)))
    _box_mask = np.stack(tuple((row[3] for row in targets)))
    _iou_mask = np.stack(tuple((row[4] for row in targets)))
    _class_mask = np.stack(tuple((row[5] for row in targets)))

    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask


def process_batch(data, size_index):
    W, H = cfg.multi_scale_out_size[size_index]
    inp_size = cfg.multi_scale_inp_size[size_index]
    out_size = cfg.multi_scale_out_size[size_index]
    bbox_pred_np, gt_boxes, gt_classes, dontcares, iou_pred_np = data

    # net output
    hw, num_anchors, _ = bbox_pred_np.shape

    # gt
    _classes = np.zeros([hw, num_anchors, cfg.num_classes], dtype=np.float)
    _class_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _ious = np.zeros([hw, num_anchors, 1], dtype=np.float)
    _iou_mask = np.zeros([hw, num_anchors, 1], dtype=np.float)

    _boxes = np.zeros([hw, num_anchors, 4], dtype=np.float)
    _boxes[:, :, 0:2] = 0.5
    _boxes[:, :, 2:4] = 1.0
    _box_mask = np.zeros([hw, num_anchors, 1], dtype=np.float) + 0.01
    ###
    # scale pred_bbox
    anchors = np.ascontiguousarray(cfg.anchors, dtype=np.float)
    bbox_pred_np = np.expand_dims(bbox_pred_np, 0)
    bbox_np = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred_np, dtype=np.float),
        anchors,
        H, W)
    # bbox_np = (hw, num_anchors, (x1, y1, x2, y2))   range: 0 ~ 1
    bbox_np = bbox_np[0]
    bbox_np[:, :, 0::2] *= float(inp_size[0])  # rescale x
    bbox_np[:, :, 1::2] *= float(inp_size[1])  # rescale y
    ###
    # gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)
    gt_boxes_b = np.asarray(gt_boxes, dtype=np.float)
    ###
    # for each cell, compare predicted_bbox and gt_bbox
    bbox_np_b = np.reshape(bbox_np, [-1, 4])
    ious = bbox_ious(
        np.ascontiguousarray(bbox_np_b, dtype=np.float),
        np.ascontiguousarray(gt_boxes_b, dtype=np.float)
    )
    ###
    # print(ious)
    best_ious = np.max(ious, axis=1).reshape(_iou_mask.shape)
    # !
    iou_penalty = 0 - iou_pred_np[best_ious < cfg.iou_thresh]
    _iou_mask[best_ious <= cfg.iou_thresh] = cfg.noobject_scale * iou_penalty
    # !
    # locate the cell of each gt_boxe
    cell_w = float(inp_size[0]) / W
    cell_h = float(inp_size[1]) / H
    cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
    target_boxes[:, 0] = cx - np.floor(cx)  # cx
    target_boxes[:, 1] = cy - np.floor(cy)  # cy
    target_boxes[:, 2] = \
        (gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) / inp_size[0] * out_size[0]  # tw
    target_boxes[:, 3] = \
        (gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) / inp_size[1] * out_size[1]  # th

    # for each gt boxes, match the best anchor
    gt_boxes_resize = np.copy(gt_boxes_b)
    gt_boxes_resize[:, 0::2] *= (out_size[0] / float(inp_size[0]))
    gt_boxes_resize[:, 1::2] *= (out_size[1] / float(inp_size[1]))
    anchor_ious = anchor_intersections(
        anchors,
        np.ascontiguousarray(gt_boxes_resize, dtype=np.float)
    )
    anchor_inds = np.argmax(anchor_ious, axis=0)

    ious_reshaped = np.reshape(ious, [hw, num_anchors, len(cell_inds)])
    for i, cell_ind in enumerate(cell_inds):
        if cell_ind >= hw or cell_ind < 0:
            # print('cell inds size {}'.format(len(cell_inds)))
            # print('cell over {} hw {}'.format(cell_ind, hw))
            continue
        a = anchor_inds[i]

        # 0 ~ 1, should be close to 1
        iou_pred_cell_anchor = iou_pred_np[cell_ind, a, :]
        _iou_mask[cell_ind, a, :] = cfg.object_scale * (1 - iou_pred_cell_anchor)  # noqa
        # _ious[cell_ind, a, :] = anchor_ious[a, i]
        _ious[cell_ind, a, :] = ious_reshaped[cell_ind, a, i]

        _box_mask[cell_ind, a, :] = cfg.coord_scale
        target_boxes[i, 2:4] /= anchors[a]
        _boxes[cell_ind, a, :] = target_boxes[i]

        _class_mask[cell_ind, a, :] = cfg.class_scale
        _classes[cell_ind, a, gt_classes[i]] = 1.

    # _boxes[:, :, 2:4] = np.maximum(_boxes[:, :, 2:4], 0.001)
    # _boxes[:, :, 2:4] = np.log(_boxes[:, :, 2:4])

    return _boxes, _ious, _classes, _box_mask, _iou_mask, _class_mask
