from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

pixel_means=np.array([[[102.9801,115.9465,122.7717]]])
test_scales=(600,)
test_max_size=1000
def _get_image_blob(im):
    im_orig=im.astype(np.float32,copy=True)
    im_orig-=pixel_means

    im_shape=im_orig.shape
    im_size_min=np.min(im_shape[0:2])
    im_size_max=np.max(im_shape[0:2])

    processed_ims=[]
    im_scale_factors=[]
    for target_size in test_scales:
        im_scale=float(target_size)/float(im_size_min)
        if np.round(im_scale*im_size_max)>test_max_size:
            im_scale=float(test_max_size)/float(im_size_max)
        im=
