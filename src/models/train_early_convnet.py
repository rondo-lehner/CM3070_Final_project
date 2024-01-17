## Imports
import tensorflow_datasets as tfds
import tensorflow as tf

import src.data.pipelines.convnet_pipeline as convnet_pipeline
import os
import gc
import datetime
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models import early_convnet

## Parameters
# Pipeline
BATCH_SIZE_IMAGES = 1
BATCH_SIZE_PATCHES = 1
IMAGE_SIZE = 2448
PATCH_SIZE = 40
PATCH_SIZE_ANNOTATION = 2
PATCH_STRIDE = 20
SLICE_TRAIN = ':5'
SLICE_VALID = '5:8'
SLICE_TEST = '700:720'

# Training
EPOCHS = 6
CHECKPOINT_FILEPATH = os.path.join(os.getcwd(), 'models', 'ckpt', 'early_convnet', 'weights.{epoch:02d}-{batch}.ckpt')
SAVE_FREQ = 732050 # 'epoch' or integer (saves the model at end of this many batches) | Save weights after every 50 images at full resolution
CLASS_WEIGHTS = {
        0: 6.070,    # urban_land
        1: 1.,       # agriculture_land
        2: 5.559,    # rangeland
        3: 4.128,    # forest_land
        4: 15.176,   # water
        5: 9.244,    # barren_land
        6: 100.       # unknown - Note: not to scale with respect to the others but not that important for the overall classification
}

# Tensorboard
LOG_DIR = os.path.join(os.getcwd(), 'models', 'logs', 'early_convnet', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
UPDATE_FREQ = 14641 # currently updates after every image 

## TODO: Attempt to fix creeping memory consumption.
#   * try: https://stackoverflow.com/questions/53683164/keras-occupies-an-indefinitely-increasing-amount-of-memory-for-each-epoch
#   * also try: https://github.com/tensorflow/tensorflow/issues/37505#issuecomment-643685221
#   * or maybe: https://github.com/tensorflow/tensorflow/issues/37505#issuecomment-806472122

class ClearMemoryOnEndEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

## MAIN FUNCTION

def main():
    """ TODO: add description
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')

    input_pipeline = convnet_pipeline.ConvnetPipeline(
        SLICE_TRAIN,
        SLICE_VALID,
        SLICE_TEST,
        BATCH_SIZE_IMAGES,
        BATCH_SIZE_PATCHES,
        IMAGE_SIZE,
        PATCH_SIZE,
        PATCH_SIZE_ANNOTATION,
        PATCH_STRIDE
    )

    logger.info('Created input pipeline.')

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
        save_freq=SAVE_FREQ
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        update_freq=UPDATE_FREQ,
        write_images=False # True doesn't add real benefit, it appears there is a limitation with the visualisation of Conv2D weights: https://github.com/tensorflow/tensorboard/issues/2240
    )

    logger.info('Created model, starting training...')

    model.fit(
        input_pipeline.train,
        epochs=EPOCHS,
        validation_data=input_pipeline.valid,
        class_weight=CLASS_WEIGHTS,
        callbacks=[model_checkpoint_callback, tensorboard_callback, ClearMemoryOnEndEpoch()],
        workers=0
    )

    logger.info('Training completed.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

