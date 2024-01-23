## Imports
import tensorflow_datasets as tfds
import tensorflow as tf

import src.data.datasets.deep_globe_2018
import os

class FcnPipeline():
    def __init__(
        self,
        slice_train=":70%",
        slice_valid="70%:90%",
        slice_test="90%:",
        batch_size=1,
        image_size=224
        ):
        self.batch_size=batch_size
        self.image_size=image_size

        splits = self.load_processed_splits(slice_train, slice_valid, slice_test)
    
        self.train = splits[0]
        self.valid = splits[1]
        self.test = splits[2]


    ## Helper functions

    def rgb_to_index(self, image):
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

    def load_images(self, datapoint, image_size):
    
        images = tf.image.resize(datapoint['image'], (image_size, image_size))

        annotations = tf.map_fn(self.rgb_to_index, datapoint['segmentation_mask'])
        annotations = tf.image.resize(annotations, (image_size, image_size), method='nearest')

        annotations = tf.one_hot(
                annotations,
                depth = 7, # TODO: make depth configurable
                on_value = 1,
                off_value = 0,
                axis = 3
            )
        annotations = tf.squeeze(annotations, axis=4)
        
        ## Pre-process as per tf documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16 (last accessed 23.01.2024)
        images = tf.keras.applications.vgg16.preprocess_input(images)

        return images, annotations

    def load_processed_splits(self, slice_train, slice_valid, slice_test):
        (ds_train, ds_valid, ds_test), ds_info = tfds.load(
            name='deep_globe_2018',
            download=False,
            with_info=True,
            split=[f'all_images[{slice_train}]', f'all_images[{slice_valid}]', f'all_images[{slice_test}]']
        )
        train_batches = (
            ds_train
            .batch(self.batch_size)
            .map(lambda x: self.load_images(x, self.image_size), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        validation_batches = (
            ds_valid
            .batch(self.batch_size)
            .map(lambda x: self.load_images(x, self.image_size), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        test_batches = (
            ds_test
            .batch(self.batch_size)
            .map(lambda x: self.load_images(x, self.image_size), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )    
        return (train_batches, validation_batches, test_batches)