import numpy as np
import tensorflow as tf
a=np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [5,6,7]])
c=tf.less(1,2)    #return true if x<y
with tf.Session() as sess:
    print(sess.run(c))
