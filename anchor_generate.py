from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#   anchor=
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

def generate_anchors(base_size=16,ratios=[0.5,1,2],scales=2**np.arange(3,6)):
    base_anchor=np.array([1,1,base_size,base_size])-1
    ratio_anchors=_ratio_enum(base_anchor,ratios)
    anchors=np.vstack([_scale_enum(ratio_anchors[i,:],scales) for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length


def _whcenter(anchor):
    """
    return width,height,x center,y center for an anchor
    """
    w=anchor[2]-anchor[0]+1
    h=anchor[3]-anchor[1]+1
    x_center=anchor[0]+0.5*(w-1)
    y_center=anchor[1]+0.5*(h-1)
    return w,h,x_center,y_center

def _mkanchors(ws,hs,x_center,y_center):
    """
    Given a vector of width and heights around a center,output a set of anchors
    """
    ws=ws[:,np.newaxis]
    hs=hs[:,np.newaxis]
    anchors=np.hstack((x_center-0.5*(ws-1),
                       y_center-0.5*(hs-1),
                       x_center+0.5*(ws+1),
                       y_center+0.5*(hs+1)))
    return anchors

def _ratio_enum(anchor,ratios):
    """
    enumerate a set of anchors for each aspect ratio wrt an anchor
    """
    w,h,x_center,y_center=_whcenter(anchor)
    size=w*h
    size_ratios=size/ratios
    ws=np.round(np.sqrt(size_ratios))
    hs=np.round(ws*ratios)
    anchors=_mkanchors(ws,hs,x_center,y_center)
    return anchors

def _scale_enum(anchor,scales):
    """
    enumerate a set of anchors for each scale wrt an anchor.
    """
    w,h,x_center,y_center=_whcenter(anchor)
    ws=w*scales
    hs=h+scales
    anchors=_mkanchors(ws,hs,x_center,y_center)
    return anchors

if __name__=='__main__':
    import time
    t=time.time()
    a=generate_anchors()
    print(time.time()-t)
    print(a)
















