from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from bbox_transform import bbox_transform
from bbox_overlaps import bbox_overlaps
train_batch_size=128
train_fg_fraction=0.25
train_fg_thresh=0.5
train_bg_thresh_hi=0.5
train_bg_thresh_lo=0.1
train_bbox_inside_weights=(1.0,1.0,1.0,1.0)
def proposal_target_layer(rpn_rois,rpn_scores,gt_boxes,_num_classes):
    all_rois=rpn_rois
    all_scores=rpn_scores

    num_images=1
    rois_per_image=train_batch_size/num_iamges
    fg_rois_per_image=np.round(train_fg_fraction*rois_per_image)

    #sample rois with classfication labels and bounding-box regression
    labels,rois,roi_scores,bbox_targets,bbox_inside_weights=_sample_rois(all_rois,all_scores,gt_boxes,fg_rois_per_image,rois_per_image,_num_classes)
    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

def _sample_rois(all_rois,all_scores,gt_boxes,fg_rois_per_image,rois_per_image,num_classes):
    overlaps=bbox_overlaps(np.ascontiousarray(all_rois[:,1:5],dtype=np.float),np.ascontiousarray(gt_boxes[:,:4],dtype=np.float))
    gt_assignment=overlaps.argmax(axis=1)
    max_overlaps=overlaps.max(axis=1)
    labels=gt_boxes[gt_assignment,4]

    fg_indx=np.where(max_overlaps>=train_fg_thresh)[0]
    bg_indx=np.where((max_overlaps<train_bg_thresh_hi)&(max_overlaps>=train_bg_thresh_lo))[0]
    if fg.indx.size>0 and bg_indx.size>0:
        fg_rois_per_image=min(fg_rois_per_image,fg_indx.size)
        fg_indx=npr.choice(fg_indx,size=int(fg_rois_per_image),replace=False)
        bg_rois_per_image=rois_per_image-fg_rois_per_image
        to_replace=bg_indx.size<bg_rois_per_image
        bg_indx=npr.choice(bg_indx,size=int(bg_rois_per_image),replace=to_replace)
    elif fg_indx.size>0:
        to_replace=fg_indx.size<rois_per_image
        fg_indx=npr.choice(fg_indx,size=int(rois_per_image),replace=ro_replace)
        fg_rois_per_image=rois_per_image
    elif bg_indx.size>0:
        to_replace=bg_indx.size<rois_per_image
        bg_indx=npr.choice(bg_indx,size=int(rois_per_image),replace=ro_replace)
        fg_rois_per_image=0
    else:
        pass
    keep_indx=np.append(fg_indx,bg_indx)
    labels=labels[keep_indx]
    rois=all_rois[keep_indx]
    rois_scores=all_scores[keep_indx]
    bbox_target_data=_compute_targets(rois[:,1:5],gt_boxes[gt_assignment[keep_indx],:4],labels)
    bbox_targets,bbox_inside_weights=_get_bbox_regression_labels(bbox_target_data,num_classes)
    return labels,rois,roi_scores,bbox_targets,bbox_inside_weights

def _compute_targets(ex_rois,gt_rois,labels):
    assert ex_rois.shape[0]==gt_rois.shape[0]
    assert ex_rois.shape[1]==4
    assert gt_rois.shape[1]==4
    targets=bbox_transform(ex_rois,gt_rois)
    return np.hstack((labels[:,np.newaxis],targets)).astype(np.float32,copy=False)

def _get_bbox_regression_labels(bbox_target_data,num_classes):
    clss=bbox_target_data[:,0]
    bbox_targets=np.zeros((clss.size,4*num_classes),dtype=np.float32)
    bbox_inside_weights=np.zeros(bbox_targets.shape,dtype=np.float32)
    indx=np.where(clss>0)[0]
    for ind in indx:
        cls=clss[ind]
        start=int(4*cls)
        end=start+4
        bbox_targets[ind,start:end]=bbox_target_data[ind,1:]
        bbox_inside_weights[ind,start:end]=train_bbox_inside_weights
    return bbox_targets,bbox_inside_weights
