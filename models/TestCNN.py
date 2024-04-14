
import tensorflow as tf
from tensorflow.keras import models, layers
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

class SimpleCNN(BaseModel):
    '''
    Simple CNN model class, inherits from BaseModel
    '''
    def __init__(self, output_shape=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(output_shape, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
