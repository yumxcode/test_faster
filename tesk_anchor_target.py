from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from bbox_overlaps import bbox_overlaps
from bbox_transform import bbox_transform
train_rpn_negative_overlap=0.3
train_rpn_positive_overlap=0.7
train_rpn_gf_fraction=0.5
train_rpn_batchsize=256
train_rpn_bbox_inside_weights=(1.0,1.0,1.0,1.0)
train_rpn_positive_weight=-0.1

def anchor_target_layer(rpn_cls_score,gt_boxes,im_info,_feat_stride,all_anchors,num_anchors):
    A=num_anchors
    total_anchors=all_anchors.shape[0]
    K=total_anchors/num_anchors

    _allowed_border=0
    height,width=rpn_cls_score.shape[1:3]

    indx_inside = np.where((all_anchors[:, 0] >= -_allowed_border) &(all_anchors[:, 1] >= -_allowed_border) &(all_anchors[:, 2] < im_info[1] + _allowed_border) &(all_anchors[:, 3] < im_info[0] + _allowed_border))[0]
    anchors=all_anchors[indx_inside,:]

    #label:1 is positive,0 is negative, -1 is ignore
    labels=np.empty((len(indx_inside),),dtype=np.float32)
    labels.fill(-1)

    overlaps=bbox_overlaps(np.ascontiguousarray(anchors,dtype=np.float),np.ascontiguousarray(gt_boxes,dtype=np.float))
    argmax_overlaps=overlaps.argmax(axis=1)
    max_overlaps=overlaps[np.arange(len(indx_inside)),argmax_overlaps]
    gt_argmax_overlaps=overlaps.argmax(axis=0)
    gt_max_overlaps=overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    gt_argmax_overlaps=np.where(overlaps==gt_max_overlaps)[0]
    labels[max_overlaps<train_rpn_negative_overlap]=0
    labels[gt_argmax_overlaps]=1
    labels[max_overlaps>=train_rpn_positive_overlap]=1
    # subsample positive labels if we have many
    num_fg=int(train_rpn_fg_fraction*train_rpn_batchsize)
    fg_indx=np.where(labels==1)[0]
    if len(fg_indx)>num_fg:
        disable_indx=npr.choice(fg_indx,size=(len(fg_indx)-num_fg),replace=False)
        labels[disable_indx]=-1
    #subsample negative labels if we have many
    num_bg=train_rpn_batchsize-np.sum(labels==1)
    bg_indx=np.where(labels==0)[0]
    if len(bg_indx)>num_bg:
        disable_indx=npr.choice(bg_indx,size=(len(bg_indx)-num_bg),replace=False)
        labels[disable_indx]=-1
    bbox_targets=np.zeros((len(indx_inside),4),dtype=np.float32)
    bbox_targets=_compute_targets(anchors,gt_boxes[argmax_overlaps,:])
    bbox_inside_weights=np.zeros((len(indx_inside),4),dtype=np.float32)
    bbox_inside_weights[labels==1,:]=np.array(train_rpn_bbox_inside_weights)
    bbox_outside_weights=np.zeros((len(indx_inside),4),dtype=np.float32)
    if train_rpn_positive_weight<0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    bbox_outside_weights[labels==1,:]=positive_weights
    bbox_outside_weights[labels==0,:]=negative_weights

    #map up to original set of anchors
    labels=_unmap(labels,total_anchors,indx_inside,fill=-1)
    bbox_targets=_unmap(bbox_targets,total_anchors,indx_inside,fill=0)
    bbox_inside_weights=_unmap(bbox_inside_weights,total_anchors,indx_inside,fill=0)
    bbox_outside_weights=_unmap(bbox_outside_weights,total_anchors,indx_inside,fill=0)
    labels=labels.reshape((1,height,width,A)).transpose(0,3,1,2)
    labels=labels.reshape((1,1,A*height,width))
    rpn_labels=labels
    bbox_targets=bbox_targets.reshape((1,height,width,A*4))
    rpn_bbox_targets=bbox_targets
    bbox_inside_weights=bbox_inside_weights.reshape((1,height,width,A*4))
    rpn_bbox_inside_weights=bbox_inside_weights
    bbox_outside_weights=bbox_outside_weights.rshape((1,height,width,A*4))
    rpn_bbox_outside_weights=bbox_outside_weights
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights


def _unmap(data,count,indx,fill=0):
    if len(data.shape)==1:
        ret=np.empty((count,),dtype=np.float32)
        ret.fill(fill)
        ret[indx]=data
    else:
        ret=np.empty((count,)+data.shape[1:],dtype=np.float32)
        ret.fill(fill)
        ret[indx,:]=data
    return ret
    

def _compute_targets(ex_rois,gt_rois):
    assert ex_rois.shape[0]==gt_rois.shape[0]
    assert ex_rois.shape[1]==4
    assert gt_rois.shape[1]==5

    return bbox_transform(ex_rois,gt_rois[:,:4]).astype(np.float32,copy=False)
