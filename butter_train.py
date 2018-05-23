#coding:utf-8
import tensorflow as tf
from PIL import Image
import xml.dom.minidom as xml
import numpy as np
from vgg_16 import vgg16

#def demo(sess,net):
base_filename='./tesk_train/IMG_00000'
gt_boxes=np.zeros((1,5),dtype=np.float32)
dom=xml.parse(base_filename+str(1)+'.xml')
root=dom.documentElement
bbox=root.getElementsByTagName('xmin')
gt_boxes[0,0]=bbox[0].childNodes[0].data
bbox=root.getElementsByTagName('ymin')
gt_boxes[0,1]=bbox[0].childNodes[0].data
bbox=root.getElementsByTagName('xmax')
gt_boxes[0,2]=bbox[0].childNodes[0].data
bbox=root.getElementsByTagName('ymax')
gt_boxes[0,3]=bbox[0].childNodes[0].data
gt_boxes[:,4]=1
img=Image.open(base_filename+str(1)+'.jpg')
img=np.array(img)
img=img[np.newaxis,:]
blobs={}
blobs['im_info']=np.array([img.shape[0],img.shape[1]],dtype=np.float32)
blobs['data']=img
sess=tf.Session()
#built network
net=vgg16()
net.create_architecture(1,'test',True)
init=tf.global_variables_initializer()
sess.run(init)
_,scores,bbox_pred,rois=net.test_image(sess,blobs['data'],blobs['im_info'])

print(scores)
print(bbox_pred)
print(rois)
