
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os
import sys

sys.path.append('..')
from cutout.cutout import DynamicCutoutLayer

import logging
os.creatdir('logs', exist_ok=True)
from logging.handlers import RotatingFileHandler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/cutout.log', maxBytes=10*1024*1024, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class BaseModel(models.Model):
    '''
    Base Model class, should be used as Abstract class for the other models.
    The class provides the functionality to use dynamic cutout augmentation
    based on saliency maps of the input images.
    - The use_cutout flag is used to enable or disable the dynamic cutout augmentation
    - The cutout_active flag is used to enable or disable the dynamic cutout augmentation
    - The current_epoch counter is used to track the current epoch
    - The cutout_layer is the dynamic cutout layer to apply the cutout augmentation
    args:
        :use_cutout: bool, whether to use dynamic cutout augmentation or not
        :cutout_layer: DynamicCutoutLayer, the dynamic cutout layer to apply the cutout augmentation
    '''
    def __init__(self, use_cutout=False, cutout_layer:DynamicCutoutLayer=None ,*args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.use_cutout = use_cutout
        self.cutout_active = False
        self.current_epoch = 0
        if cutout_layer is None:
            cutout_layer = DynamicCutoutLayer(mask_size=16, cutout_prob=0.8, shape='square')
        self.cutout_layer = cutout_layer

    @tf.function
    def train_step(self, data):
        '''
        Training step, apply dynamic cutout augmentation if enabled
        args:
            :data: tuple, batch of the input image and label pair as a tuple
        return:
            :dict: the loss and the metrics of the model
        '''
        # use the saliency cutout layer for dynamic cutout augmentation based on saliency maps of the input images
        x, y = data
        if self.use_cutout and self.current_epoch > 2 and not self.cutout_active:
            self.enanble_cutout()
        if self.cutout_active and hasattr(self, 'cutout_layer'):
            x = self.cutout_layer((x, y), self, training=True)

        # continue with the training step
        return super().train_step((x, y))
    
    def enanble_cutout(self, cutout_layer:DynamicCutoutLayer=None):
        '''
        Enable the dynamic cutout augmentation, can override the current cutout layer with new configuration
        args:
            :cutout_layer: DynamicCutoutLayer, the dynamic cutout layer to apply the cutout augmentation
        '''
        self.cutout_active = True
        if cutout_layer is not None:
            self.cutout_layer = cutout_layer

    def diable_cutout(self):
        self.cutout_active = False

    def build_graph(self, input_shape=(None, 32, 32, 3)):
        x = layers.Input(shape=input_shape)
        return models.Model(inputs=[x], outputs=self.call(x))
    
    def on_epoch_end(self):
        self.current_epoch += 1  # increment the epoch counter

    # def on_epoch_begin(self):
    #     pass

    def on_train_end(self):
        self.current_epoch = 0  # reset the epoch counter
    
    # def on_train_begin(self):
    #     pass