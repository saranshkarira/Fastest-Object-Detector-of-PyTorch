import os
import numpy as np
import pickle
import json

class_map = {'Weapon': [0], 'Vehicle': [1], 'Building': [2], 'Person': [3]}

# given precision and recall, calculates AP in voc syle using  either 07 metric or the new metric


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# makes a pickle file from annotation file if not present for faster parsing as eval is needed multiple times, then matches GTs with predictions to calculate precision and recall to calculate AP


def voc_eval(detpath, annopath, classname, cachedir, ovthresh=0.5, use_07_metric=False):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    if not os.path.exists(cachefile):

        with open(annopath) as opener:
            annots = json.load(opener)
        recs = []
        for i in annots:
            gt_boxes = []
            gt_classes = []
            imagenames = []
            for k, v in i.iteritems():
                if k == 'Var1':
                    img = v.split('/')[-1]

                elif k == classname:

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
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # filter by class:
    class_recs = {}

    # seq = np.asarray(range(nd))
    # off = np.ones(nd)
    # tpfn = seq+off
    tpfn = 0
    for imagename in imagenames:

        bbox = np.array(list(filter(lambda x: recs[imagename]['gt_boxes'] if recs[imagename]['gt_classes'] == class_map[classname] else False, recs[imagename]['gt_boxes'])))

        det = [False] * bbox.shape(0)
        tpfn += sum(~np.ones(bbox.shape(0)))
        class_recs[imagename] = {'bbox': bbox, 'det': det}
    # Open detection, and annotation and then calculate precision recall
    detfile = detpath.format(classname)

    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go through dets and mark TPs and FPs using IoU threshold
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # Union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1
                    R['det'][jmax] = 1

                else:
                    fp[d] = 1

            else:
                fp[d] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(tpfn)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

    else:
        rec = -1
        prec = -1
        ap = -1

    return rec, prec, ap
