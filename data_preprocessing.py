import tensorflow as tf

from cutout.utils import random_cutout

def prepare_dataset(dataset:tf.data.Dataset, kind:str, batch_size=32):
    autotuner = tf.data.experimental.AUTOTUNE

    dataset = dataset.map(preprocess_CIFAR10, num_parallel_calls=autotuner)
    
    # if test val ( not training data ) do not augment the images
    if kind == "train":
        print('train_ds')
        dataset = dataset.map(augment_image, num_parallel_calls=autotuner)
        # dataset = dataset.map(lambda x, y: (random_cutout(x, training=True), y))
    
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotuner)
    return dataset

def preprocess_CIFAR10(image, label):
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.one_hot(label, 10)
    return image, label

def augment_image(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    # Randomly apply mirroring
    image = tf.image.random_flip_left_right(image)
    # Randomly apply rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    image = random_cutout(image, 20, p=1, shape='circle')

    return image, label
