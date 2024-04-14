import tensorflow as tf
from cutout.cutout import RandomCutoutLayer

def prepare_dataset(dataset:tf.data.Dataset, kind:str, batch_size=32, 
                    cutout_layer:RandomCutoutLayer=None):
    '''
    Prepare the dataset for training, validation, or testing by applying preprocessing steps
    and augmentations
    args:
        :dataset: tf.data.Dataset, the input dataset
        :kind: str, the kind of the dataset, either 'train', 'val', or 'test'
        :batch_size: int, the batch size
        :cutout_layer: RandomCutoutLayer, the cutout layer to apply cutout augmentation
    return:
        :tf.data.Dataset: the prepared dataset
    '''
    autotuner = tf.data.experimental.AUTOTUNE

    dataset = dataset.map(preprocess_CIFAR10, num_parallel_calls=autotuner)
    
    # if test val ( not training data ) do not augment the images
    if kind == "train":
        dataset = dataset.map(lambda img, label:
                                (apply_random_augmentations(img), label),
                                num_parallel_calls=autotuner)
        if cutout_layer:
            dataset = dataset.map(lambda img, label: 
                                  (cutout_layer(img, training=True), label),
                                  num_parallel_calls=autotuner)
    
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotuner)
    return dataset

def preprocess_CIFAR10(image, label):
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.one_hot(label, 10)
    return image, label

def apply_random_augmentations(image, max_num_augmentations=4, seed=None, fixed_augmentations=None):

    def resize_crop(img):
        n, m = img.shape[:2]
        scale = 1.25
        n_scaled = tf.cast(n*scale,tf.int32)
        m_scaled = tf.cast(m*scale,tf.int32)
        return tf.image.random_crop(
            tf.image.resize_with_crop_or_pad(img,
            n_scaled, m_scaled),
            size=image.shape)
    
    def apply_augmentation(img, idx):
        # Use tf.switch_case to select and apply an augmentation
        return tf.switch_case(idx, {
            0: lambda: resize_crop(img),
            1: lambda: tf.image.random_flip_left_right(img),
            2: lambda: tf.image.random_flip_up_down(img),
            3: lambda: tf.image.random_brightness(img, max_delta=0.1),
            4: lambda: tf.image.random_contrast(img, lower=0.1, upper=0.2),
            5: lambda: tf.image.random_saturation(img, lower=0.1, upper=0.2),
            6: lambda: tf.image.random_hue(img, max_delta=0.1), 
            },
            default=lambda: img)
    
    # augmentation_names = {0: 'resize_crop', 1: 'flip_left_right', 2: 'flip_up_down',
    #                       3: 'brightness', 4: 'contrast', 5: 'saturation', 6: 'hue'}
    augmentations = {
        0: lambda x: resize_crop(x),
        1: lambda x: tf.image.random_flip_left_right(x),
        2: lambda x: tf.image.random_flip_up_down(x),
        3: lambda x: tf.image.random_brightness(x, max_delta=0.1),
        4: lambda x: tf.image.random_contrast(x, lower=0.1, upper=0.2),
        5: lambda x: tf.image.random_saturation(x, lower=0.1, upper=0.2),
        6: lambda x: tf.image.random_hue(x, max_delta=0.1),   
    }

    augmentation_idx = tf.random.shuffle(tf.range(len(augmentations)))
    augmentation_idx = tf.cast(augmentation_idx, tf.int32)
    num_to_apply = tf.random.uniform(shape=[], minval=1, maxval=max_num_augmentations, dtype=tf.int32)

    choosen_augmentations = augmentation_idx[:num_to_apply]
    if fixed_augmentations:
        choosen_augmentations = tf.constant(fixed_augmentations)

    # initial image to start with
    augmented_img = image
    # apply `num_to_apply` augmentations to the image
    for i in choosen_augmentations:
        augmented_img = apply_augmentation(augmented_img, i)
    return augmented_img
    # for aug in choosen_augmentations:
    #     image = augmentations[aug.numpy()](image)
    # return image