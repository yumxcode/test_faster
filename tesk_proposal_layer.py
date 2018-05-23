from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from bbox_transform import bbox_transform_inv,clip_boxes

train_rpn_pre_nms_topN=12000
train_rpn_post_nms_topN=2000
test_rpn_pre_nms_topN=6000
test_rpn_post_nms_topN=300
rpn_nms_thresh=0.7

def proposal_layer(rpn_cls_prob,rpn_bbox_pred,im_info,_feat_stride,anchors,num_anchors,is_training):
    scores=rpn_cls_prob[:,:,:,num_anchors:]
    rpn_bbox_pred=rpn_bbox_pred.reshape((-1,4))
    scores=scores.reshape((-1,1))
    proposals=bbox_transform_inv(anchors,rpn_bbox_pred)
    proposals=clip_boxes(proposals,im_info[:2])
    
    # pick the top region proposals:
    order=scores.ravel().argsort()[::-1]
    if is_training:
        order=order[:train_rpn_pre_nms_topN]
    else:
        order=order[:test_rpn_pre_nms_topN]
    proposals=proposals[order,:]
    scores=scores[order]
    keep=nms(np.hstack((proposals,scores)),rpn_nms_thresh)
    if is_training:
        keep=keep[:train_rpn_nms_post_topN]
    else:
        keep=keep[:test_rpn_nms_post_topN]
    proposals=proposals[keep,:]
    scores=scores[keep]
    #only support single image as input:
    batch_indx=np.zeros((proposals.shape[0],1),dtype=np.float32)
    blob=np.hstack((batch_indx,proposals.astype(np.float32,copy=False)))
    return blob,scores



