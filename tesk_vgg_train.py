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
blobs['gt_boxes']=gt_boxes

if __name__=='__main__':
    image=tf.placeholder(tf.float32.shape=img.shape)
    net=vgg16()
    net=net.test(image,True)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict={image:img}
        print(sess.run(net,feed_dict=feed_dict).shape)
