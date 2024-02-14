## Imports
import tensorflow as tf

import src.data.pipelines.unet_pipeline as unet_pipeline
import os
import gc
import datetime
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models import unet

## Pipeline
SPLIT_TRAIN = ":10"
SPLIT_VALID = "10:15"
SPLIT_TEST = "0:15"
BATCH_SIZE = 1
IMAGE_SIZE = 228    # value model-compatible value to 224
BORDER = 92

## Training
EPOCHS = 5
LEARNING_RATE = 1e-3
MOMENTUM = 0.99
LOAD_WEIGHTS = False
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = 100//BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = 10 // BATCH_SIZE
VALIDATION_STEPS = 5 // BATCH_SIZE
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'models', 'ckpt', 'unet')
CHECKPOINT_FILEPATH = os.path.join(CHECKPOINT_DIR, '{epoch:02d}-{val_loss:.2f}.ckpt')

## Tensorboard
LOG_DIR = os.path.join(os.getcwd(), 'models', 'logs', 'unet')
UPDATE_FREQ = 'batch'

def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')

    (train, validation, test) = unet_pipeline.getUNetPipeline(
        SPLIT_TRAIN,
        SPLIT_VALID,
        SPLIT_TEST,
        BATCH_SIZE,
        IMAGE_SIZE,
        BORDER
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        save_weights_only=True,
        monitor='val_loss',
        save_freq='epoch'
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            update_freq=UPDATE_FREQ,
            write_images=False # True doesn't add real benefit, it appears there is a limitation with the visualisation of Conv2D weights: https://github.com/tensorflow/tensorboard/issues/2240
        )

    # TODO: Implement optimizer and loss as per paper
    optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model = unet.get_UNet(input_shape=(412, 412, 3))
    model.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics = [tf.keras.metrics.MeanIoU(num_classes=7, name='mIoU', sparse_y_true=False, sparse_y_pred=False)])

    if LOAD_WEIGHTS:
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        model.load_weights(latest)

    model.fit(
        train,
        epochs=EPOCHS, 
        validation_data=validation, 
        callbacks=[tensorboard_callback, model_checkpoint_callback],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS
        )

    # TODO: implement model.fit()

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