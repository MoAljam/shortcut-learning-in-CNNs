
import tensorflow as tf
from tensorflow.keras import models, layers, Model
import numpy as np
import os
import sys

from models.BaseModel import BaseModel
# sys.path.append('..')
# from cutout.cutout import DynamicCutoutLayer

import logging
from logging.handlers import RotatingFileHandler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/cutout.log', maxBytes=10*1024*1024, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def resnet_block(input_layer, filters, kernel_size, strides=(1, 1), activation='relu'):
    '''
    ResNet block
    args:
        :input_layer: tf.Tensor, the input tensor
        :filters: int, the number of filters
        :kernel_size: int, the size of the kernel
        :strides: tuple, the strides of the convolution
        :activation: str, the activation function
    return:
        :tf.Tensor: the output tensor
    '''
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

class ResNet(BaseModel):
    '''
    ResNet model class, inherits from BaseModel
    - The model is built based on the ResNet architecture
    - The model can be either ResNet18 or ResNet34
    args:
        :input_shape: tuple, the shape of the input image
        :num_classes: int, the number of classes
        :type: str, the type of the ResNet model, either 'resnet18' or 'resnet34'
    '''
    def __init__(self, input_shape, num_classes, type='resnet18'):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.type = type
        self._build_resnet(input_shape, num_classes, type)

    def _build_resnet(self, input_shape, num_classes, type):
        '''
        Build the ResNet model based on the ResNet architecture type
        '''
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        if type == 'resnet18':
            num_blocks_list = [2, 2, 2, 2]
            filters_list = [64, 128, 256, 512]
        elif type == 'resnet34':
            num_blocks_list = [3, 4, 6, 3]
            filters_list = [64, 128, 256, 512]
        else:
            raise ValueError('ResNet type not supported')

        for i, num_blocks in enumerate(num_blocks_list):
            for j in range(num_blocks):
                strides = (1, 1)
                if i > 0 and j == 0:
                    strides = (2, 2)
                if j == 0:
                    shortcut = layers.Conv2D(filters_list[i], 1, strides=strides, padding='same')(x)
                    shortcut = layers.BatchNormalization()(shortcut)
                    x = resnet_block(x, filters_list[i], 3, strides=strides)
                    x = layers.add([x, shortcut])
                else:
                    x = resnet_block(x, filters_list[i], 3)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)


# def build_resnet(input_shape, num_classes, type='resnet18'):
#     inputs = tf.keras.Input(shape=input_shape)

#     x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

#     if type == 'resnet18':
#         num_blocks_list = [2, 2, 2, 2]
#         filters_list = [64, 128, 256, 512]
#     elif type == 'resnet34':
#         num_blocks_list = [3, 4, 6, 3]
#         filters_list = [64, 128, 256, 512]
#     else:
#         raise ValueError('ResNet type not supported')

#     for i, num_blocks in enumerate(num_blocks_list):
#         for j in range(num_blocks):
#             strides = (1, 1)
#             if i > 0 and j == 0:
#                 strides = (2, 2)
#             if j == 0:
#                 shortcut = layers.Conv2D(filters_list[i], 1, strides=strides, padding='same')(x)
#                 shortcut = layers.BatchNormalization()(shortcut)
#                 x = resnet_block(x, filters_list[i], 3, strides=strides)
#                 x = layers.add([x, shortcut])
#             else:
#                 x = resnet_block(x, filters_list[i], 3)

#     x = layers.GlobalAveragePooling2D()(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs, outputs)
#     return model

# ResNet18 = build_resnet(input_shape=(32, 32, 3), num_classes=10, type='resnet18')
# ResNet34 = build_resnet(input_shape=(32, 32, 3), num_classes=10, type='resnet34')
