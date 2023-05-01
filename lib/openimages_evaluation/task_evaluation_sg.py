"""
Written by Ji Zhang, 2019
Some functions are adapted from Rowan Zellers
Original source:
https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
"""
import os
import numpy as np
import logging
from six.moves import cPickle as pickle
import json
import csv
from tqdm import tqdm

from functools import reduce
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.openimages_evaluation.ap_eval_rel import ap_eval, prepare_mAP_dets
from lib.pytorch_misc import intersect_2d, argsort_desc

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


def eval_rel_results(all_results, topk=100, do_val=True, do_vis=False):

    print('topk: ', topk)

    eval_ap = True
    eval_per_img = True #should be False
    prd_k = 2
    
    if eval_per_img:
        recalls = {1: [], 5: [], 10: [], 20: [], 50: [], 100: [], 200: [], 400: []}
    else:
        recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0, 200: 0, 400: 0}
        if do_val:
            all_gt_cnt = 0

    # if do_special:
    #     special_img_f = open("/home/jiz/projects/100_img_special_set.txt", "r")
    #     special_imgs = special_img_f.readlines()
    #     special_imgs = [img[:-1] for img in special_imgs]
    #     special_img_set = set(special_imgs)
    #     logger.info('Special images len: {}'.format(len(special_img_set)))
    
    topk_dets = []
    for im_i, res in enumerate(tqdm(all_results)):

        # if do_special:
        #     img_id = res['image'].split('/')[-1].split('.')[0]
        #     if img_id not in special_img_set:
        #         continue
        
        # in oi_all_rel some images have no dets
        if res['prd_scores'] is None:
            det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
            det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
            det_labels_s_top = np.zeros(0, dtype=np.int32)
            det_labels_p_top = np.zeros(0, dtype=np.int32)
            det_labels_o_top = np.zeros(0, dtype=np.int32)
            det_scores_top = np.zeros(0, dtype=np.float32)
            
            det_scores_top_vis = np.zeros(0, dtype=np.float32)
            if 'prd_scores_bias' in res:
                det_scores_top_bias = np.zeros(0, dtype=np.float32)
            if 'prd_scores_spt' in res:
                det_scores_top_spt = np.zeros(0, dtype=np.float32)
        else:
            det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
            det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
            det_labels_sbj = res['sbj_labels']  # (#num_rel,)
            det_labels_obj = res['obj_labels']  # (#num_rel,)
            det_scores_sbj = res['sbj_scores']  # (#num_rel,)
            det_scores_obj = res['obj_scores']  # (#num_rel,)
            if 'prd_scores_ttl' in res:
                det_scores_prd = res['prd_scores_ttl'] #TODO[:, 1:]
            else:
                det_scores_prd = res['prd_scores'] #TODO[:, 1:]

            det_labels_prd = np.argsort(-det_scores_prd, axis=1)
            det_scores_prd = -np.sort(-det_scores_prd, axis=1)

            det_scores_so = det_scores_sbj * det_scores_obj
            det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k]

            det_scores_inds = argsort_desc(det_scores_spo)[:topk]
            det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_boxes_so_top = np.hstack(
                (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
            det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_labels_spo_top = np.vstack(
                (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()

            # filter out bad relationships
            cand_inds = np.where(det_scores_top > 1e-08)[0]
            det_boxes_so_top = det_boxes_so_top[cand_inds]
            det_labels_spo_top = det_labels_spo_top[cand_inds]
            det_scores_top = det_scores_top[cand_inds]

            det_scores_vis = res['prd_scores'] #TODO[:, 1:][:, 1:]
            for i in range(det_labels_prd.shape[0]):
                det_scores_vis[i] = det_scores_vis[i][det_labels_prd[i]]
            det_scores_vis = det_scores_vis[:, :prd_k]
            det_scores_top_vis = det_scores_vis[det_scores_inds[:, 0], det_scores_inds[:, 1]]
            det_scores_top_vis = det_scores_top_vis[cand_inds]
            if 'prd_scores_bias' in res:
                det_scores_bias = res['prd_scores_bias'][:, 1:]
                for i in range(det_labels_prd.shape[0]):
                    det_scores_bias[i] = det_scores_bias[i][det_labels_prd[i]]
                det_scores_bias = det_scores_bias[:, :prd_k]
                det_scores_top_bias = det_scores_bias[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                det_scores_top_bias = det_scores_top_bias[cand_inds]
            if 'prd_scores_spt' in res:
                det_scores_spt = res['prd_scores_spt'][:, 1:]
                for i in range(det_labels_prd.shape[0]):
                    det_scores_spt[i] = det_scores_spt[i][det_labels_prd[i]]
                det_scores_spt = det_scores_spt[:, :prd_k]
                det_scores_top_spt = det_scores_spt[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                det_scores_top_spt = det_scores_top_spt[cand_inds]
            
            det_boxes_s_top = det_boxes_so_top[:, :4]
            det_boxes_o_top = det_boxes_so_top[:, 4:]
            det_labels_s_top = det_labels_spo_top[:, 0]
            det_labels_p_top = det_labels_spo_top[:, 1]
            det_labels_o_top = det_labels_spo_top[:, 2]
            
        topk_dets.append(dict(image=res['image'],
                              det_boxes_s_top=det_boxes_s_top,
                              det_boxes_o_top=det_boxes_o_top,
                              det_labels_s_top=det_labels_s_top,
                              det_labels_p_top=det_labels_p_top,
                              det_labels_o_top=det_labels_o_top,
                              det_scores_top=det_scores_top))
        topk_dets[-1]['det_scores_top_vis'] = det_scores_top_vis
        if 'prd_scores_bias' in res:
            topk_dets[-1]['det_scores_top_bias'] = det_scores_top_bias
        if 'prd_scores_spt' in res:
            topk_dets[-1]['det_scores_top_spt'] = det_scores_top_spt
        if do_vis:
            topk_dets[-1].update(dict(blob_conv=res['blob_conv'],
                                      blob_conv_prd=res['blob_conv_prd']))

        if do_val:
            gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
            gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
            gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
            gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
            gt_labels_prd = res['gt_prd_labels']  # (#num_gt,)
            gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
            gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
            # Compute recall. It's most efficient to match once and then do recall after
            # det_boxes_so_top is (#num_rel, 8)
            # det_labels_spo_top is (#num_rel, 3)
            pred_to_gt = _compute_pred_matches(
                gt_labels_spo, det_labels_spo_top,
                gt_boxes_so, det_boxes_so_top)
            if eval_per_img:
                for k in recalls:
                    if len(pred_to_gt):
                        match = reduce(np.union1d, pred_to_gt[:k])
                    else:
                        match = []
                    rec_i = float(len(match)) / float(gt_labels_spo.shape[0] + 1e-12)  # in case there is no gt
                    recalls[k].append(rec_i)
            else:    
                all_gt_cnt += gt_labels_spo.shape[0]
                for k in recalls:
                    if len(pred_to_gt):
                        match = reduce(np.union1d, pred_to_gt[:k])
                    else:
                        match = []
                    recalls[k] += len(match)
            
            topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                      gt_boxes_obj=gt_boxes_obj,
                                      gt_labels_sbj=gt_labels_sbj,
                                      gt_labels_obj=gt_labels_obj,
                                      gt_labels_prd=gt_labels_prd))

    if do_val:
        if eval_per_img:
            for k, v in recalls.items():
                recalls[k] = np.mean(v)
        else:
            for k in recalls:
                recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
        excel_str = print_stats(recalls)      
        if eval_ap:
            # prepare dets for each class
            logger.info('Preparing dets for mAP...')
            cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(topk_dets, 31)
            all_npos = sum(npos)
            with open('data/vg/rel.json') as f: #TODO relative path!!!!!!!!!!!
                rel_prd_cats = json.load(f)['rel_categories']

            rel_mAP = 0.
            w_rel_mAP = 0.
            ap_str = ''
            for c in range(31):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True)
                weighted_ap = ap * float(npos[c]) / float(all_npos)
                w_rel_mAP += weighted_ap
                rel_mAP += ap
                ap_str += '{:.2f}, '.format(100 * ap)
                print('rel AP for class {}: {:.2f} ({:.6f})'.format(rel_prd_cats[c], 100 * ap, float(npos[c]) / float(all_npos)))
            rel_mAP /= 31.
            print('weighted rel mAP: {:.2f}'.format(100 * w_rel_mAP))
            excel_str += ap_str

            phr_mAP = 0.
            w_phr_mAP = 0.
            ap_str = ''
            for c in range(31):
                rec, prec, ap = ap_eval(cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False)
                weighted_ap = ap * float(npos[c]) / float(all_npos)
                w_phr_mAP += weighted_ap
                phr_mAP += ap
                ap_str += '{:.2f}, '.format(100 * ap)
                print('phr AP for class {}: {:.2f} ({:.6f})'.format(rel_prd_cats[c], 100 * ap, float(npos[c]) / float(all_npos)))
            phr_mAP /= 31.
            print('weighted phr mAP: {:.2f}'.format(100 * w_phr_mAP))
            excel_str += ap_str
            
            # total: 0.4 x rel_mAP + 0.2 x R@50 + 0.4 x phr_mAP
            final_score = 0.4 * rel_mAP + 0.2 * recalls[50] + 0.4 * phr_mAP
            
            # total: 0.4 x w_rel_mAP + 0.2 x R@50 + 0.4 x w_phr_mAP
            w_final_score = 0.4 * w_rel_mAP + 0.2 * recalls[50] + 0.4 * w_phr_mAP
            print('weighted final_score: {:.2f}'.format(100 * w_final_score))
            
            # get excel friendly string
            # excel_str = '{:.2f}, {:.2f}, {:.2f}, {:.2f}, '.format(100 * recalls[50], 100 * w_rel_mAP, 100 * w_phr_mAP, 100 * w_final_score) + excel_str
            # print('Excel-friendly format:')
            # print(excel_str.strip()[:-1])
    
    #print('Saving topk dets...')
    # topk_dets_f = os.path.join(output_dir, 'rel_detections_topk_{}.pkl'.format(topk))
    # with open(topk_dets_f, 'wb') as f:
    #     pickle.dump(topk_dets, f, pickle.HIGHEST_PROTOCOL)
    logger.info('topk_dets size: {}'.format(len(topk_dets)))
    print('Done.')


def print_stats(recalls):
    # print('====================== ' + 'sgdet' + ' ============================')
    k_str = ''
    for k in recalls.keys():
        if k == 50:
            continue
        k_str += '{}\t'.format(k)
    v_str = ''
    for k, v in recalls.items():
        print('R@%i: %.2f' % (k, 100 * v))
        if k == 50:
            continue
        v_str += '{:.2f}, '.format(100 * v)
    return v_str


# This function is adapted from Rowan Zellers' code:
# https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
# Modified for this project to work with PyTorch v0.4
def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: Do y
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            gt_box_union = gt_box_union.astype(dtype=np.float32, copy=False)
            box_union = box_union.astype(dtype=np.float32, copy=False)
            inds = bbox_overlaps(gt_box_union[None], 
                                 box_union = box_union)[0] >= iou_thresh

        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
