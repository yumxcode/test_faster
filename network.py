from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses


import numpy as np
import tensorflow as tf
class Network(object):
    def __init__(self):
        self._predictions={}
        self._losses={}
        self._anchor_targets={}
        self._proposal_targets={}
        self._layers={}
        self._gt_image=None
        #not very understand:
        self._act_summaries=[]
        self._score_summaries={}
        self._train_summaries=[]
        self._event_summaries={}
        self._variables_to_fix={}
    def _add_gt_image(self):
        image=self._image+
