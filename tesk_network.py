from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from tesk_proposal_layer import proposal_layer
from tesk_anchor_target import anchor_target_layer
from tesk_proposal_target import proposal_target_layer
from anchor_generate import generate_anchors_pre
class Network(object):
    def __init__(self):
        self._train_batch_size=128
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None

    def _softmax_layer(self,bottom,name):

        if name.startswith('rpn_cls_prob_reshape'):
            input_shape=tf.shape(bottom)
            bottom_reshaped=tf.reshape(bottom,[-1,input_shape[-1]])
            reshaped_score=tf.nn.softmax(bottom_reshaped,name=name)
            return tf.reshape(reshaped_score,input_shape)
        return tf.nn.softmax(bottom,name=name)


    def _reshape_layer(self,bottom,num_dim,name):
        input_shape=tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            to_caffe=tf.transpose(bottom,[0,3,1,2])
            reshaped=tf.reshape(to_caffe,tf.concat(axis=0,values=[[1,num_dim,-1],[input_shape[2]]]))
            to_tf=tf.transpose(reshaped,[0,2,3,1])
            return to_tf

    def _anchor_component(self):
        with tf.variable_scope('Anchor_component') as scope:
            height=tf.to_int32(tf.ceil(self._im_info[0]/np.float32(self._feat_stride)))
            width=tf.to_int32(tf.ceil(self._im_info[1]/np.float32(self._feat_stride)))
            anchors,anchor_length=tf.py_func(generate_anchors_pre,[height,width,self._feat_stride,self._anchor_scales,self._anchor_ratios],[tf.float32,tf.int32],name='generate_anchors')
            anchors.set_shape([None,4])
            anchor_length.set_shape([])
            self._anchors=anchors
            self._anchor_length=anchor_length

    def _anchor_target_layer(self,rpn_cls_score,name):
        with tf.variable_scope(name) as scope:
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights=tf.py_func(anchor_target_layer,[rpn_cls_score,self._gt_boxes,self._im_info,self._feat_stride,self._anchors,self._num_anchors],[tf.float32,tf.float32,tf.float32,tf.float32],name='anchor_target')
            rpn_labels.set_shape([1,1,None,None])
            rpn_bbox_targets.set_shape([1,None,None,self._num_anchors*4])
            rpn_bbox_inside_weights.set_shape([1,None,None,self._num_anchors*4])
            rpn_bbox_outside_weights.set_shape([1,None,None,self._num_anchors*4])

            rpn_labels=tf.to_int32(rpn_labels,name='to_int32')
            self._anchor_targets['rpn_labels']=rpn_labels
            self._anchor_targets['rpn_bbox_targets']=rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights']=rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights']=rpn_bbox_outside_weights
        return rpn_labels




    def _crop_pool_layer(self, bottom, rois, name):
        pooling_size=7
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
        return slim.max_pool2d(crops, [2, 2], padding='SAME')



    def _build_network(self,is_training=True):
        initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01)
        initializer_bbox=tf.truncated_normal_initializer(mean=0.0,stddev=0.001)
        net_conv=self._image_to_head(is_training)
        with tf.variable_scope('build_network') as scope:
            self._anchor_component()
            rois=self._region_proposal(net_conv,is_training,initializer)
            #region of interest pooling
            pool5=self._crop_pool_layer(net_conv,rois,'pool5')
            fc7=self._head_to_tail(pool5,is_training)
            with tf.variable_scope('region_generate') as scope:
                cls_prob,bbox_pred=self._region_classification(fc7,is_training,initializer,initializer_bbox)
        return rois,cls_prob,bbox_pred



    def _region_proposal(self,net_conv,is_training,initializer):
        rpn=slim.conv2d(net_conv,512,[3,3],trainable=is_training,weights_initializer=initializer,scope='rpn_conv/3_3')
        rpn_cls_score=slim.conv2d(rpn,self._num_anchors*2,[1,1],trainable=is_training,weights_initializer=initializer,padding='VALID',activation_fn=None,scope='rpn_cls_score')
        #change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_bbox_pred=slim.conv2d(rpn,self._num_anchors*4,[1,1],trainable=is_training,weights_initializer=initializer,padding='VALID',activation_fn=None,scope='rpn_bbox_pred')
        if is_training:
            rois,roi_scores=self._proposal_layer(rpn_cls_prob,rpn_bbox_pred,is_training,'rois')
            rpn_labels=self._anchor_target_layer(rpn_cls_score,'anchor')
            rois,_=self._proposal_target_layer(rois,roi_scores,'rpn_rois')
        else:
            rois,_=self._proposal_layer(rpn_cls_prob,rpn_bbox_pred,'rois')
        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        return rois




    def _proposal_layer(self,rpn_cls_prob,rpn_bbox_pred,is_training,name):
        with tf.variable_scope(name) as scope:
            rois,rpn_scores=tf.py_func(proposal_layer,[rpn_cls_prob,rpn_bbox_pred,self._im_info,self._feat_stride,self._anchors,self._num_anchors,is_training],[tf.float32,tf.float32],name='proposal')
            rois.set_shape([None,5])
            rpn_scores.set_shape([None,1])
        return rois,rpn_scores


    def _proposal_target_layer(self,rois,roi_scores,name):
        with tf.variable_scope(name) as scope:
            rois,roi_scores,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights=tf.py_func(proposal_target_layer,[rois,roi_scores,self._gt_boxes,self._num_classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32],name='proposal_target')
            rois.set_shape([self._train_batch_size,5])
            roi_scores.set_shape([self._train_batch_size])
            labels.set_shape([self._train_batch_size,1])
            bbox_targets.set_shape([self._train_batch_size,self._num_classes*4])
            bbox_inside_weights.set_shape([self._train_batch_size,self._num_classes*4])
            bbox_outside_weights.set_shape([self._train_batch_size,self._num_classes*4])
            self._proposal_targets['rois']=rois
            self._proposal_targets['labels']=tf.to_int32(labels,name='to_int32')
            self._proposal_targets['bbox_targets']=bbox_targets
            self._proposal_targets['bbox_inside_weights']=bbox_inside_weights
            self._proposal_targets['bbox_outside_weights']=bbox_outside_weights
            return rois,roi_scores



    def _smooth_l1_loss(self,bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights,sigma=1.0,dim=[1]):
        sigma_2=sigma**2
        box_diff=bbox_pred-bbox_targets
        in_box_diff=bbox_inside_weights*box_diff
        abs_in_box_diff=tf.abs(in_box_diff)
        smoothl1_sign=tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff,1./sigma_2)))
        in_loss_box=tf.pow(in_box_diff,2)*(sigma_2 /2.)*smoothl1_sign+(abs_in_box_diff-(0.5/sigma_2))*(1. - smoothl1_sign)
        out_loss_box=bbox_outside_weights*in_loss_box
        loss_box=tf.reduce_mean(tf.reduce_sum(out_loss_box,axis=dim))
        return loss_box
        
    def _add_losses(self,sigma_rpn=3.0):
        with tf.variable_scope('Loss_'+self._tag) as scope:
            #rpn, class loss:
            rpn_cls_score=tf.reshape(self._predictions['rpn_cls_score_reshape'],[-1,2])
            rpn_label=tf.reshape(self._anchor_targets['rpn_labels'],[-1])
            rpn_select=tf.where(tf.not_equal(rpn_label,-1))
            rpn_cls_score=tf.reshape(tf.gather(rpn_cls_score,rpn_select),[-1,2])
            rpn_label=tf.reshape(tf.gather(rpn_label,rpn_select),[-1])
            rpn_cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,labels=rpn_label))

            # rpn, bbox_loss:
            rpn_bbox_pred=self._predictions['rpn_bbox_pred']
            rpn_bbox_targets=self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights=self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights=self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box=self._smooth_l1_loss(rpn_bbox_pred,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights,sigma=sigma_rpn,dim=[1,2,3])


            #rcnn,class loss:
            cls_score=self._predictions['cls_score']
            label=tf.reshape(self._proposal_targets['labels'],[-1])
            cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score,labels=label))

            #rcnn,bbox loss:
            bbox_pred=self._predictions['bbox_pred']
            bbox_targets=self._proposal_targets['bbox_targets']
            bbox_inside_weights=self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights=self._proposal_targets['bbox_outside_weights']
            loss_box=self._smooth_l1_loss(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights)

            self._losses['cross_entropy']=cross_entropy
            self._losses['loss_box']=loss_box
            self._losses['rpn_cross_entropy']=rpn_cross_entropy
            self._losses['rpn_loss_box']=rpn_loss_box

            loss=cross_entropy+loss_box+rpn_cross_entropy+rpn_loss_box
            regularization_loss=tf.add_n(tf.losses.get_regularization_losses(),'regu')
            self._losses['total_loss']=loss+regularization_loss
            
        return loss


    def _region_classification(self,fc7,is_training,initializer,initializer_bbox):
        cls_score=slim.fully_connected(fc7,self._num_classes,weights_initializer=initializer,trainable=is_training,activation_fn=None,scope='cls_score')
        cls_prob=self._softmax_layer(cls_score,'cls_prob')
        cls_pred=tf.argmax(cls_score,axis=1,name='cls_pred')
        bbox_pred=slim.fully_connected(fc7,self._num_classes*4,weights_initializer=initializer_bbox,trainable=is_training,activation_fn=None,scope='bbox_pred')

        self._predictions['cls_score']=cls_score
        self._predictions['cls_pred']=cls_pred
        self._predictions['cls_prob']=cls_prob
        self._predictions['bbox_pred']=bbox_pred

        return cls_prob,bbox_pred

    def create_architecture(self,num_classes,tag,is_training,anchor_scales=(8,16,32),anchor_ratios=(0.5,1,2)):
        self._image=tf.placeholder(tf.float32,shape=[1,None,None,3])
        self._im_info=tf.placeholder(tf.float32,shape=[2])
        self._gt_boxes=tf.placeholder(tf.float32,shape=[None,5])
        self._tag=tag

        self._num_classes=num_classes
        self._anchor_scales=anchor_scales
        self._num_scales=len(anchor_scales)

        self._anchor_ratios=anchor_ratios
        self._num_ratios=len(anchor_ratios)

        self._num_anchors=self._num_scales*self._num_ratios
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001)
        biases_regularizer=tf.no_regularizer
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], weights_regularizer=weights_regularizer,biases_regularizer=biases_regularizer, biases_initializer=tf.constant_initializer(0.0)):
            rois,cls_prob,bbox_pred=self._build_network(is_training)
        if not is_training:
            stds = np.tile(np.array((0.1,0.1,0.2,0.2)), (self._num_classes))
            means = np.tile(np.array((0.0,0.0,0.0,0.0)), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],self._losses['rpn_loss_box'],self._losses['cross_entropy'],self._losses['loss_box'],self._losses['total_loss'],train_op],feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def extract_head(self,sess,image):
        feed_dict={self._image:image}
        feat=sess.run(self.layers['head'],feed_dict=feed_dict)
        return feat

    def test_image(self,sess,image,im_info):
        feed_dict={self._image:image,self._im_info:im_info}
        cls_score,cls_prob,bbox_pred,rois=sess.run([self._predictions['cls_score'],self._predictions['cls_prob'],self._predictions['bbox_pred'],self._predictions['rois']],feed_dict=feed_dict)
        return cls_score,cls_prob,bbox_pred,rois











