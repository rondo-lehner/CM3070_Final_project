## Imports
import tensorflow_datasets as tfds
import tensorflow as tf

import src.data.datasets.deep_globe_2018
import os
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models import early_convnet

## Parameters
# Pipeline
BATCH_SIZE_IMAGES = 1
BATCH_SIZE_PATCHES = 1
IMAGE_SIZE = 612
PATCH_SIZE = 40
PATCH_SIZE_ANNOTATION = 2
PATCH_STRIDE = 32
SLICE_TRAIN = ':70'
SLICE_VALID = '70:90'
SLICE_TEST = '90:100'

# Training
EPOCHS = 4
CHECKPOINT_FILEPATH = os.path.join(os.getcwd(), 'models', 'ckpt', 'early_convnet', 'weights.{epoch:02d}-{val_loss:.2f}.ckpt')
CLASS_WEIGHTS = {
        0: 6.070,    # urban_land
        1: 1.,       # agriculture_land
        2: 5.559,    # rangeland
        3: 4.128,    # forest_land
        4: 15.176,   # water
        5: 9.244,    # barren_land
        6: 100.       # unknown - Note: not to scale with respect to the others but not that important for the overall classification
}

## Helper functions

def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def rgb_to_index(image):
    palette = [
        [0, 255, 255],   # urban_land
        [255, 255, 0],   # agriculture_land
        [255, 0, 255],   # rangeland
        [0, 255, 0],     # forest_land
        [0, 0, 255],     # water
        [255, 255, 255], # barren_land
        [0, 0, 0]        # unknown
    ]
    
    one_hot_map = []
    for colour in palette:
        class_map = tf.reduce_all(tf.equal(image, colour), axis=-1)
        one_hot_map.append(class_map)
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.uint8)
    indexed = tf.math.argmax(one_hot_map, axis=2)
    indexed = tf.cast(indexed, dtype=tf.uint8)
    indexed = tf.expand_dims(indexed, -1)

    return indexed

def reduce_mode(tensor):
    y, idx, count = tf.unique_with_counts(tensor)
    mode = y[tf.argmax(count)]
    return mode

def load_patches_labels(datapoint, image_size, patch_size, patch_size_annotation, stride):
    crop_fraction = patch_size_annotation / patch_size
    
    images = tf.image.resize(datapoint['image'], (image_size, image_size))
    img_patches = tf.image.extract_patches(
        images = images,
        sizes = [1, patch_size, patch_size, 1],
        strides = [1, stride, stride, 1],
        rates = [1, 1, 1, 1],
        padding = 'VALID'
    )
    img_patches_flat = tf.reshape(img_patches, shape=(-1, patch_size, patch_size, 3))

    annotations = tf.map_fn(rgb_to_index, datapoint['segmentation_mask'])
    annotations = tf.image.resize(annotations, (image_size, image_size), method='nearest')

    ann_patches = tf.image.extract_patches(
        images = annotations,
        sizes = [1, patch_size, patch_size, 1],
        strides = [1, stride, stride, 1],
        rates = [1, 1, 1, 1],
        padding = 'VALID'
    )
    ann_patches_flat = tf.reshape(ann_patches, shape=(-1, patch_size, patch_size, 1))
    central_pixels = tf.image.central_crop(ann_patches_flat, crop_fraction)
    dim = tf.reduce_prod(tf.shape(central_pixels)[1:])
    central_pixels = tf.reshape(central_pixels, [-1, dim])

    # pixel_category_idx = tf.map_fn(reduce_mode, central_pixels)
    # print(central_pixels)
    pixel_category_idx = tf.reduce_max(central_pixels, axis=1) # reduce_mode is probably preferred but I chose a simpler implementation

    img_patches_flat = normalize(img_patches_flat)
    pixel_category_one_hot = tf.one_hot(
        pixel_category_idx,
        depth = 7, # TODO: make depth configurable
        on_value = 1,
        off_value = 0
    )

    pixel_category_one_hot = tf.expand_dims(pixel_category_one_hot, axis=1)
    pixel_category_one_hot = tf.expand_dims(pixel_category_one_hot, axis=1)

    return img_patches_flat, pixel_category_one_hot

def load_processed_splits():
    (ds_train, ds_valid, ds_test), ds_info = tfds.load(
        name='deep_globe_2018',
        download=False,
        with_info=True,
        split=[f'all_images[{SLICE_TRAIN}]', f'all_images[{SLICE_VALID}]', f'all_images[{SLICE_TEST}]']
    )
    train_batches = (
        ds_train
        .batch(BATCH_SIZE_IMAGES)
        .map(lambda x: load_patches_labels(x, IMAGE_SIZE, PATCH_SIZE, PATCH_SIZE_ANNOTATION, PATCH_STRIDE), num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch() # Flatten the batches for training
        .batch(BATCH_SIZE_PATCHES) # Rebatch patches as desired
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    validation_batches = (
        ds_valid
        .batch(BATCH_SIZE_IMAGES)
        .map(lambda x: load_patches_labels(x, IMAGE_SIZE, PATCH_SIZE, PATCH_SIZE_ANNOTATION, PATCH_STRIDE), num_parallel_calls=tf.data.AUTOTUNE)
        .unbatch() # Flatten the batches for training
        .batch(BATCH_SIZE_PATCHES) # Rebatch patches as desired
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_batches = (
        ds_test
        .batch(BATCH_SIZE_IMAGES)
        .map(lambda x: load_patches_labels(x, IMAGE_SIZE, PATCH_SIZE, PATCH_SIZE_ANNOTATION, PATCH_STRIDE), num_parallel_calls=tf.data.AUTOTUNE)
        ## unbatching not required for testing
        # .unbatch() # Flatten the batches for training
        # .batch(batch_size_patches) # Rebatch patches as desired
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return (train_batches, validation_batches, test_batches)

## MAIN FUNCTION

def main():
    """ TODO: add description
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')

    split_train, split_valid, split_test = load_processed_splits()

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    model = early_convnet.EarlyConvnet()
    model.build((None, PATCH_SIZE, PATCH_SIZE, 3))
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = ['mse', tf.keras.metrics.CategoricalCrossentropy(name='cce')]
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        save_weights_only=True,
        monitor='val_cce',
        save_freq='epoch'
    )

    model.fit(
        split_train,
        epochs=EPOCHS,
        validation_data=split_valid,
        class_weight=CLASS_WEIGHTS,
        callbacks=[model_checkpoint_callback]
    )


    logger.info('splits loaded.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

