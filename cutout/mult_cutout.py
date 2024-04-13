# dataset wrapper for multiple cutout
import tensorflow as tf
# from tensorflow.data import Dataset
# from tensorflow.keras import datasets
import numpy as np
# PROTOTYPE

class CutOut(tf.data.Dataset):
    def __new__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        y = tf.keras.utils.to_categorical(1, 10)
        return x, y
    
class SaliencyCutoutLayer(tf.keras.layers.Layer):
    def __init__(self, model, mask_size=16, **kwargs):
        super(SaliencyCutoutLayer, self).__init__(**kwargs)
        self.model = model
        self.mask_size = mask_size
        # trainable flag to control augmentation
        self.augment_flag = self.add_weight(name='augment_flag', 
                                            shape=(),
                                            initializer=tf.constant_initializer(0),
                                            trainable=False,
                                            dtype=tf.int32)

    def call(self, inputs):
        # check if augmentation should be applied
        if self.augment_flag == 0:
            return inputs
        
        # compute saliency maps and apply cutout 
        
        return inputs

    def enable_augmentation(self):
        self.augment_flag.assign(1)

    def disable_augmentation(self):
        self.augment_flag.assign(0)