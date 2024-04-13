import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os

# PROTOTYPE of simple CNN model
class Net(models.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def build_graph(self):
        x = layers.Input(shape=(32, 32, 3))
        return models.Model(inputs=[x], outputs=self.call(x))
