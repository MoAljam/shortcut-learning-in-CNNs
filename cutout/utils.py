import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

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
   
@tf.function
def compute_saliency_map(model, input_image, target):
    '''
    Compute the saliency map of the input image
    args:
        :model: tf.keras.Model, the model to compute the saliency map
        :input_image: tf.Tensor, the input image
        :target: tf.Tensor, the target label
    return:
        :tf.Tensor: the saliency map
    '''
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        predictions = model(input_image, training=True)
        loss = model.compiled_loss(target, predictions)
    
    gradients = tape.gradient(loss, input_image)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)
    # saliency_map = tf.reduce_mean(tf.abs(gradients), axis=-1)
    return saliency_map

@tf.function
def max_saliency_coords(saliency_map, filter_size=3):
    '''
    Compute the coordinates of the maximum activation in the saliency map
    args:
        :saliency_map: tf.Tensor, the saliency map
        :filter_size: int, the size of the filter to smooth the saliency map
    return:
        :tf.Tensor: the coordinates of the maximum activation
    '''
    # return x,y # coordinates of the max activation
    # smooth the saliency map with a filter gasussian filter
    saliency_map = tf.expand_dims(saliency_map, axis=-1)
    saliency_map = tf.nn.conv2d(saliency_map,
                                tf.ones((filter_size, filter_size, 1, 1), dtype=tf.float32) / tf.cast(filter_size**2, tf.float32),
                                strides=1, padding='SAME')
    saliency_map = tf.squeeze(saliency_map, axis=-1)
    # max_coords = tf.stack(tf.unravel_index(tf.argmax(saliency_map[0], axis=None), saliency_map[0].shape), axis=0)
    # compute the indices of the max activation
    flat_index = tf.argmax(tf.reshape(saliency_map, [saliency_map.shape[0], -1]), axis=1)
    max_coords = tf.stack([
        flat_index // saliency_map.shape[2],  # y coordinates
        flat_index % saliency_map.shape[2]   # x coordinates
    ], axis=1)
    return max_coords

@tf.function
def zero_out(img, coords, mask_size, shape='square'):
    '''
    Zero out a region of the input image(single image) at given coordinates in 2d
    args:
        :img: tf.Tensor, the input image
        :coords: tf.Tensor, the coordinates of the center of the region to zero out
        :mask_size: int, the size of the region to zero out
        :shape: str, the shape of the region to zero out, either 'square' or 'circle'
    return:
        :tf.Tensor: the input image with the region zeroed out
    '''
    img_type = img.dtype
    mid_y, mid_x = coords[0], coords[1]
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

@tf.function
def zero_out_batched(input, coords, mask_size, shape='square'):
    '''
    Zero out a region of the input image(batched) at given coordinates in 2d (batched)
    args:
        :input: tf.Tensor, the input image
        :coords: tf.Tensor, the coordinates of the center of the region to zero out
        :mask_size: int, the size of the region to zero out
        :shape: str, the shape of the region to zero out, either 'square' or 'circle'
    return:
        :tf.Tensor: the input image with the region zeroed out
    '''

    # apply zero_out to each element of the batch
    elems = (input, coords)
    output = tf.map_fn(
        lambda x: zero_out(x[0], x[1], mask_size, shape),
        elems,
        dtype=input.dtype
    )
    return output



# plotting functions ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def plot_progress(history, test_results, model_name, title='Training Progress'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} {title}')
    # accuracy
    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].plot(test_results['test_accuracy'], label='test')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    # loss
    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].plot(test_results['test_loss'], label='test')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_attention(input:tf.data.Dataset, model:tf.keras.Model, cutout_layer, label_names, num_samples=4, title='Data Samples'):
    # Define the number of columns and rows
    num_columns = 3  # Since each sample has 3 images side by side
    num_rows = num_samples  # Each sample needs its own row
    fig_width = num_columns * 6
    fig_height = num_rows * 5

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
    fig.suptitle(title)

    # Flatten the array of axes if it's not already flat (needed if num_rows = 1)
    axes = axes.flatten()

    for i, (image, label) in enumerate(input.take(num_samples)):
        # Calculate the column start index
        col_start = i * num_columns

        # Process the image and predictions
        image_saliency = cutout_layer.get_network_attention((image, label), model)
        image_cutout = cutout_layer((image, label), model, training=True)
        
        # Display images
        axes[col_start].imshow((image[0] + 1) / 2)
        axes[col_start+1].imshow((image_saliency[0] + 1) / 2)
        axes[col_start+2].imshow((image_cutout[0] + 1) / 2)
        
        # Display predictions and labels
        pred = model.predict(image)
        true_label = label_names[np.argmax(label[0])]
        pred_label = label_names[np.argmax(pred[0])]
        label_text = f"True: {true_label}\nPredicted: {pred_label}"
        
        axes[col_start].set_title(label_text)
        axes[col_start+1].set_title('Saliency')
        axes[col_start+2].set_title('Cutout')
        
        for j in range(3):
            axes[col_start + j].axis('off')

    plt.tight_layout()
    plt.show()

    return fig, axes