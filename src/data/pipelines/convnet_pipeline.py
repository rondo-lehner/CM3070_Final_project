## Imports
import tensorflow_datasets as tfds
import tensorflow as tf

import src.data.datasets.deep_globe_2018
import os

class ConvnetPipeline():
    def __init__(
        self,
        slice_train=":70%",
        slice_valid="70%:90%",
        slice_test="90%:",
        batch_size_images=1,
        batch_size_patches=1,
        image_size=2448,
        patch_size=40,
        patch_size_annotation=2,
        patch_stride=4
        ):
        self.batch_size_images=batch_size_images
        self.batch_size_patches=batch_size_patches
        self.image_size=image_size
        self.patch_size=patch_size
        self.patch_size_annotation=patch_size_annotation
        self.patch_stride=patch_stride

        splits = self.load_processed_splits(slice_train, slice_valid, slice_test)
    
        self.train = splits[0]
        self.valid = splits[1]
        self.test = splits[2]


    ## Helper functions

    def normalize(self, input_image):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image

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

    def reduce_mode(tensor):
        y, idx, count = tf.unique_with_counts(tensor)
        mode = y[tf.argmax(count)]
        return mode

    def load_patches_labels(self, datapoint, image_size, patch_size, patch_size_annotation, stride):
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

        annotations = tf.map_fn(self.rgb_to_index, datapoint['segmentation_mask'])
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

        img_patches_flat = self.normalize(img_patches_flat)
        pixel_category_one_hot = tf.one_hot(
            pixel_category_idx,
            depth = 7, # TODO: make depth configurable
            on_value = 1,
            off_value = 0
        )

        pixel_category_one_hot = tf.expand_dims(pixel_category_one_hot, axis=1)
        pixel_category_one_hot = tf.expand_dims(pixel_category_one_hot, axis=1)

        return img_patches_flat, pixel_category_one_hot

    def load_processed_splits(self, slice_train, slice_valid, slice_test):
        (ds_train, ds_valid, ds_test), ds_info = tfds.load(
            name='deep_globe_2018',
            download=False,
            with_info=True,
            split=[f'all_images[{slice_train}]', f'all_images[{slice_valid}]', f'all_images[{slice_test}]']
        )
        train_batches = (
            ds_train
            .batch(self.batch_size_images)
            .map(lambda x: self.load_patches_labels(x, self.image_size, self.patch_size, self.patch_size_annotation, self.patch_stride), num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch() # Flatten the batches for training
            .batch(self.batch_size_patches) # Rebatch patches as desired
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        validation_batches = (
            ds_valid
            .batch(self.batch_size_images)
            .map(lambda x: self.load_patches_labels(x, self.image_size, self.patch_size, self.patch_size_annotation, self.patch_stride), num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch() # Flatten the batches for training
            .batch(self.batch_size_patches) # Rebatch patches as desired
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        test_batches = (
            ds_test
            .batch(self.batch_size_images)
            .map(lambda x: self.load_patches_labels(x, self.image_size, self.patch_size, self.patch_size_annotation, self.patch_stride), num_parallel_calls=tf.data.AUTOTUNE)
            ## unbatching not required for testing
            # .unbatch() # Flatten the batches for training
            # .batch(batch_size_patches) # Rebatch patches as desired
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return (train_batches, validation_batches, test_batches)