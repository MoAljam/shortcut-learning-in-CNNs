import tensorflow as tf
import numpy as np

def zero_out(img, coords, mask_size, shape='square'):
    img_type = img.dtype
    mid_y, mid_x = coords
    n, m, channels = img.shape

    padded_cutout = None
    if shape == 'square':
        cutout_area = tf.ones([mask_size, mask_size, channels], dtype=img_type)
        padded_cutout = tf.pad(cutout_area, [
            [(mid_y - mask_size // 2), (n - (mid_y + mask_size // 2))],
            [(mid_x - mask_size // 2), (m - (mid_x + mask_size // 2))],
            [0, 0]
        ], mode='CONSTANT', constant_values=0)
    elif shape == 'circle':
        # Create a circular mask
        diameter = mask_size
        x = tf.range(m, dtype=img_type) - tf.cast(mid_x, img_type)
        y = tf.range(n, dtype=img_type) - tf.cast(mid_y, img_type)
        X, Y = tf.meshgrid(x, y)
        cutout_area = tf.sqrt(tf.square(X) + tf.square(Y)) < diameter / 2 # circle coords
        cutout_area = tf.cast(cutout_area, img_type)
        padded_cutout = tf.expand_dims(cutout_area, axis=-1)

    return img * (1 - padded_cutout)


def random_cutout(img, mask_size, p=0.5, shape='square'):
    """
    Applies a cutout augmentation to the image.
    
    Args:
    img (tf.Tensor): Tensor of the image.
    mask_size (int): The size of the square or circle to cut out.
    p (float): Probability of applying the augmentation.
    shape (str): 'square' or 'circle' to specify the shape of the cutout.

    Returns:
    tf.Tensor: The augmented image.
    """
    if tf.random.uniform([]) > p:
        return img

    n, m, channels = img.shape
    mid_x = tf.cast(tf.random.uniform([], mask_size // 2, m - mask_size // 2), tf.int32)
    mid_y = tf.cast(tf.random.uniform([], mask_size // 2, n - mask_size // 2), tf.int32)

    return zero_out(img, (mid_y, mid_x), mask_size, shape=shape)