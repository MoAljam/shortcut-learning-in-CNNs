
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os
import sys

from cutout.utils import zero_out_batched, zero_out, compute_saliency_map, max_saliency_coords

import logging
from logging.handlers import RotatingFileHandler
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/cutout.log', maxBytes=10*1024*1024, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


DYNAMIC_CUTOUT_PROB = 0.8

class DynamicCutoutLayer(tf.keras.layers.Layer):
    '''
    Dynamic Cutout Layer
    - Compute the saliency map of the input image
    - Apply cutout (zero out) to the input image based on the saliency map
    - Only apply cutout if the prediction is correct
    - The cutout probability is set to 0.8 by default
    - The mask size is set to 16 by default
    - The shape of the cutout mask is set to 'square' by default, alternatively 'circle' can be used
    args:
        :mask_size: int, the size of the cutout mask
        :cutout_prob: float, the probability of applying cutout
        :shape: str, the shape of the cutout mask, either 'square' or 'circle'
        :saliency_method: function, the method to compute the saliency map
        :blur_method: function, the method to apply cutout (zero out) to the input image
    '''
    def __init__(self, mask_size, cutout_prob=DYNAMIC_CUTOUT_PROB ,shape='square',
                 saliency_method=compute_saliency_map, blur_method=zero_out_batched, **kwargs):
        super(DynamicCutoutLayer, self).__init__(**kwargs)
        self.mask_size = mask_size
        self.cutout_prob = cutout_prob
        self.shape = shape
        self.saliency_method = saliency_method
        self.blur_method = blur_method


    @tf.function
    def call(self, input, model, training=True):
        '''
        Apply dynamic cutout to the input image
        args:
            :input: tuple, batch of the input image and label pair as a tuple 
            :model: tf.keras.Model, the model to compute the saliency map
            :training: bool, whether the model is training or not
        return:
            :tf.Tensor: the input image with dynamic cutout applied
        '''
        x, y = input
        if not training or tf.random.uniform([]) > DYNAMIC_CUTOUT_PROB:
            return x
        # compute saliency maps and apply cutout
        # only if the prediction is correct
        # get the gradients of the model with respect to the input image
        saliency_map = self.saliency_method(model, x, y)
        # logger.info(f'saliency map shape: {saliency_map.shape}')
        # get the max activation coordinates
        max_coords = max_saliency_coords(saliency_map, filter_size=3)
        # logger.info(f'max coords shape: {max_coords.shape}')
        # apply cutout (zero out) to the input image
        x = self.blur_method(x, max_coords, mask_size=self.mask_size, shape=self.shape)
        return x
    
    @tf.function
    def get_network_attention(self, input, model):
        '''
        Get the network attention based on the saliency map
        args:
            :input: tf.Tensor, the input image
            :model: tf.keras.Model, the model to compute the saliency map
        return:
            :tf.Tensor: the network attention
        '''
        x, y = input[0], input[1]
        saliency_map = self.saliency_method(model, x, y)
        return saliency_map
    
    def get_config(self):
        return {
            'mask_size': self.mask_size,
            'cutout_prob': self.cutout_prob,
            'shape': self.shape,
            'saliency_method': self.saliency_method,
            'blur_method': self.blur_method
        }
    

class RandomCutoutLayer(tf.keras.layers.Layer):
    '''
    Random Cutout Layer
    - Apply cutout (zero out) to the input image at random locations
    - The cutout probability is set to 0.5 by default
    - The mask size is set to 16 by default
    - The shape of the cutout mask is set to 'square' by default, alternatively 'circle' can be used
    args:
        :mask_size: int, the size of the cutout mask
        :cutout_prob: float, the probability of applying cutout
        :shape: str, the shape of the cutout mask, either 'square' or 'circle'
    '''
    def __init__(self, mask_size, cutout_prob=0.5, shape='square'):
        super(RandomCutoutLayer, self).__init__()
        self.mask_size = mask_size
        self.cutout_prob = cutout_prob
        self.shape = shape

    @tf.function
    def call(self, input, training=False):
        '''
        Apply random cutout to the input image
        args:
            :input: tf.Tensor, the input image
            :training: bool, whether the model is training or not
        return:
            :tf.Tensor: the input image with random cutout applied
        '''
        if training:
            if tf.random.uniform([]) > self.cutout_prob:
                return input

            n, m, channels = input.shape
            mid_x = tf.cast(tf.random.uniform([], self.mask_size // 2, 
                                                m - self.mask_size // 2), tf.int32)
            mid_y = tf.cast(tf.random.uniform([], self.mask_size // 2,
                                                n - self.mask_size // 2), tf.int32)

            return zero_out(input, (mid_y, mid_x), self.mask_size, shape=self.shape)
        
        return input

    def get_config(self):
        return {
            'mask_size': self.mask_size,
            'cutout_prob': self.cutout_prob,
            'shape': self.shape
        }