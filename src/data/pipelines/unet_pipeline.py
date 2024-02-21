## Imports
import tensorflow_datasets as tfds
import tensorflow as tf

import src.data.datasets.deep_globe_2018
import os


## Helper functions
@tf.function
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

@tf.function
def load_images(datapoint, image_size, border, test=False):

    images = tf.image.resize(datapoint['image'], (image_size, image_size))

    annotations = tf.map_fn(rgb_to_index, datapoint['segmentation_mask'])
    annotations = tf.image.resize(annotations, (image_size, image_size), method='nearest')

    annotations = tf.one_hot(
            annotations,
            depth = 7,
            on_value = 1,
            off_value = 0,
            axis = 3
        )
    annotations = tf.squeeze(annotations, axis=4)

    paddings = tf.constant([[0, 0], [border, border], [border, border], [0, 0]], dtype=tf.int32)

    # normalize -1 to 1
    images = (tf.cast(images, tf.float32) / 127.5) - 1.0

    images = tf.raw_ops.MirrorPad(
        input=images, paddings=paddings, mode='REFLECT', name=None
    )

    if test:
        return images, annotations, datapoint['file_name']
    else:
        return images, annotations

## Main function
def getUNetPipeline(
    slice_train=":70%",
    slice_valid="70%:90%",
    slice_test="90%:",
    batch_size=1,
    image_size=228,
    border=92
    ):

    (ds_train, ds_valid, ds_test), ds_info = tfds.load(
        name='deep_globe_2018',
        download=False,
        with_info=True,
        split=[f'all_images[{slice_train}]', f'all_images[{slice_valid}]', f'all_images[{slice_test}]']
    )
    train_batches = (
        ds_train
        .cache()
        .shuffle(buffer_size=803, reshuffle_each_iteration=True) # in theory the full dataset
        .batch(batch_size)
        .map(lambda x: load_images(x, image_size, border), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    validation_batches = (
        ds_valid
        .cache()
        .batch(batch_size)
        .map(lambda x: load_images(x, image_size, border), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_batches = (
        ds_test
        .batch(batch_size)
        .map(lambda x: load_images(x, image_size, border, True), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )    
    return (train_batches, validation_batches, test_batches)