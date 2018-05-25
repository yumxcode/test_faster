import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from tesk_network import Network
class vgg16(Network):
    def __init__(self):
        self._scope='vgg_16'
    #built network before fc layer
                                                    
    def _image_to_head(self,is_training,reuse=None):
        with tf.variable_scope(self._scope,self._scope,reuse=None):
            net=slim.repeat(self._image,2,slim.conv2d,64,[3,3],trainable=False,scope='conv1')   #2 layer
            net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool1')
            net=slim.repeat(net,2,slim.conv2d,128,[3,3],trainable=False,scope='conv2')
            net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool2')
            net=slim.repeat(net,3,slim.conv2d,256,[3,3],trainable=is_training,scope='conv3')
            net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool3')
            net=slim.repeat(net,3,slim.conv2d,512,[3,3],trainable=is_training,scope='conv4')
            net=slim.max_pool2d(net,[2,2],padding='SAME',scope='pool4')
            net=slim.repeat(net,3,slim.conv2d,512,[3,3],trainable=is_training,scope='conv5')
            return net
            
        
        #built fc layer network:
    def _head_to_tail(self,pool5,is_training,reuse=None):
        with tf.variable_scope(self._scope,self._scope,reuse=None):
            pool5_flat=slim.flatten(pool5,scope='flatten')
            fc6=slim.fully_connected(pool5_flat,4096,scope='fc6')
            if is_training:
                fc6=slim.dropout(fc6,keep_prob=0.5,is_training=True,scope='dropout6')
            fc7=slim.fully_connected(fc6,4096,scope='fc7')
            if is_training:
                fc7=slim.dropout(fc7,keep_prob=0.5,is_training=True,scope='dropout7')
            return fc7


    def test(self,image,is_training):
        self._image=image
        net=self._image_to_head(is_training)
        net=self._head_to_tail(net,is_training)
        return net
