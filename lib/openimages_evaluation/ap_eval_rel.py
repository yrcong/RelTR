# Adapted from Detectron.pytorch/lib/datasets/voc_eval.py for
# this project by Ji Zhang, 2019
#-----------------------------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""relationship AP evaluation code."""

from six.moves import cPickle as pickle
import logging
import numpy as np
import os
from tqdm import tqdm
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps

logger = logging.getLogger(__name__)

def boxes_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    xmin = np.minimum(boxes1[:, 0], boxes2[:, 0])
    ymin = np.minimum(boxes1[:, 1], boxes2[:, 1])
    xmax = np.maximum(boxes1[:, 2], boxes2[:, 2])
    ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])
    return np.vstack((xmin, ymin, xmax, ymax)).transpose()

def prepare_mAP_dets(topk_dets, cls_num):
    cls_image_ids = [[] for _ in range(cls_num)]
    cls_dets = [{'confidence': np.empty(0),
                 'BB_s': np.empty((0, 4)),
                 'BB_o': np.empty((0, 4)),
                 'BB_r': np.empty((0, 4)),
                 'LBL_s': np.empty(0),
                 'LBL_o': np.empty(0)} for _ in range(cls_num)]
    cls_gts = [{} for _ in range(cls_num)]
    npos = [0 for _ in range(cls_num)]
    for dets in tqdm(topk_dets):
        image_id = dets['image'].split('/')[-1].split('.')[0]
        sbj_boxes = dets['det_boxes_s_top']
        obj_boxes = dets['det_boxes_o_top']
        rel_boxes = boxes_union(sbj_boxes, obj_boxes)
        sbj_labels = dets['det_labels_s_top']
        obj_labels = dets['det_labels_o_top']
        prd_labels = dets['det_labels_p_top']
        det_scores = dets['det_scores_top']
        gt_boxes_sbj = dets['gt_boxes_sbj']
        gt_boxes_obj = dets['gt_boxes_obj']
        gt_boxes_rel = boxes_union(gt_boxes_sbj, gt_boxes_obj)
        gt_labels_sbj = dets['gt_labels_sbj']
        gt_labels_prd = dets['gt_labels_prd']
        gt_labels_obj = dets['gt_labels_obj']
        for c in range(cls_num):
            cls_inds = np.where(prd_labels == c)[0]
            # logger.info(cls_inds)
            if len(cls_inds):
                cls_sbj_boxes = sbj_boxes[cls_inds]
                cls_obj_boxes = obj_boxes[cls_inds]
                cls_rel_boxes = rel_boxes[cls_inds]
                cls_sbj_labels = sbj_labels[cls_inds]
                cls_obj_labels = obj_labels[cls_inds]
                cls_det_scores = det_scores[cls_inds]
                cls_dets[c]['confidence'] = np.concatenate((cls_dets[c]['confidence'], cls_det_scores))
                cls_dets[c]['BB_s'] = np.concatenate((cls_dets[c]['BB_s'], cls_sbj_boxes), 0)
                cls_dets[c]['BB_o'] = np.concatenate((cls_dets[c]['BB_o'], cls_obj_boxes), 0)
                cls_dets[c]['BB_r'] = np.concatenate((cls_dets[c]['BB_r'], cls_rel_boxes), 0)
                cls_dets[c]['LBL_s'] = np.concatenate((cls_dets[c]['LBL_s'], cls_sbj_labels))
                cls_dets[c]['LBL_o'] = np.concatenate((cls_dets[c]['LBL_o'], cls_obj_labels))
                cls_image_ids[c] += [image_id] * len(cls_inds)
            cls_gt_inds = np.where(gt_labels_prd == c)[0]
            cls_gt_boxes_sbj = gt_boxes_sbj[cls_gt_inds]
            cls_gt_boxes_obj = gt_boxes_obj[cls_gt_inds]
            cls_gt_boxes_rel = gt_boxes_rel[cls_gt_inds]
            cls_gt_labels_sbj = gt_labels_sbj[cls_gt_inds]
            cls_gt_labels_obj = gt_labels_obj[cls_gt_inds]
            cls_gt_num = len(cls_gt_inds)
            det = [False] * cls_gt_num
            npos[c] = npos[c] + cls_gt_num
            cls_gts[c][image_id] = {'gt_boxes_sbj': cls_gt_boxes_sbj,
                                    'gt_boxes_obj': cls_gt_boxes_obj,
                                    'gt_boxes_rel': cls_gt_boxes_rel,
                                    'gt_labels_sbj': cls_gt_labels_sbj,
                                    'gt_labels_obj': cls_gt_labels_obj,
                                    'gt_num': cls_gt_num,
                                    'det': det}
    return cls_image_ids, cls_dets, cls_gts, npos


def get_ap(rec, prec):
    """Compute AP given precision and recall.
    """
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


def ap_eval(image_ids,
             dets,
             gts,
             npos,
             rel_or_phr=True,
             ovthresh=0.5):
    """
    Top level function that does the relationship AP evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    
    confidence = dets['confidence']
    BB_s = dets['BB_s']
    BB_o = dets['BB_o']
    BB_r = dets['BB_r']
    LBL_s = dets['LBL_s']
    LBL_o = dets['LBL_o']

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB_s = BB_s[sorted_ind, :]
    BB_o = BB_o[sorted_ind, :]
    BB_r = BB_r[sorted_ind, :]
    LBL_s = LBL_s[sorted_ind]
    LBL_o = LBL_o[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    gts_visited = {k: [False] * v['gt_num'] for k, v in gts.items()}
    for d in range(nd):
        R = gts[image_ids[d]]
        visited = gts_visited[image_ids[d]]
        bb_s = BB_s[d, :].astype(float)
        bb_o = BB_o[d, :].astype(float)
        bb_r = BB_r[d, :].astype(float)
        lbl_s = LBL_s[d]
        lbl_o = LBL_o[d]
        ovmax = -np.inf
        BBGT_s = R['gt_boxes_sbj'].astype(float)
        BBGT_o = R['gt_boxes_obj'].astype(float)
        BBGT_r = R['gt_boxes_rel'].astype(float)
        LBLGT_s = R['gt_labels_sbj']
        LBLGT_o = R['gt_labels_obj']
        if BBGT_s.size > 0:
            valid_mask = np.logical_and(LBLGT_s == lbl_s, LBLGT_o == lbl_o)
            if valid_mask.any():
                if rel_or_phr:  # means it is evaluating relationships
                    overlaps_s = bbox_overlaps(
                        bb_s[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_s.astype(dtype=np.float32, copy=False))[0]
                    overlaps_o = bbox_overlaps(
                        bb_o[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_o.astype(dtype=np.float32, copy=False))[0]
                    overlaps = np.minimum(overlaps_s, overlaps_o)
                else:
                    overlaps = bbox_overlaps(
                        bb_r[None, :].astype(dtype=np.float32, copy=False),
                        BBGT_r.astype(dtype=np.float32, copy=False))[0]
                overlaps *= valid_mask
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            else:
                ovmax = 0.
                jmax = -1

        if ovmax > ovthresh:
            if not visited[jmax]:
                tp[d] = 1.
                visited[jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) + 1e-12)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = get_ap(rec, prec)

    return rec, prec, ap
