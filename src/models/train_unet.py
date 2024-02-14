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
SPLIT_TRAIN = ":70%"
SPLIT_VALID = "70%:85%"
SPLIT_TEST = "85%:"
BATCH_SIZE = 32
IMAGE_SIZE = 224

## Training
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LOAD_WEIGHTS = False
MODEL = 'fcn_8s'               # fcn_32s, fcn_16s, fcn_8s
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = 100//BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = 563 // BATCH_SIZE
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'models', 'ckpt', MODEL)
CHECKPOINT_FILEPATH = os.path.join(CHECKPOINT_DIR, '{epoch:02d}-{batch}.ckpt')

## Tensorboard
LOG_DIR = os.path.join(os.getcwd(), 'models', 'logs', MODEL)
UPDATE_FREQ = 1

def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')

    (train, validation, test) = fcn_pipeline.getFCNPipeline(
        SPLIT_TRAIN,
        SPLIT_VALID,
        SPLIT_TEST,
        BATCH_SIZE,
        IMAGE_SIZE
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_FILEPATH,
        save_weights_only=True,
        monitor='val_cce',
        save_freq='epoch'
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            update_freq=UPDATE_FREQ,
            write_images=False # True doesn't add real benefit, it appears there is a limitation with the visualisation of Conv2D weights: https://github.com/tensorflow/tensorboard/issues/2240
        )

    # TODO: Implement optimizer and loss as per paper
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    if LOAD_WEIGHTS:
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        model.load_weights(latest)
    # fcn_32s.compile(optimizer=optimizer, loss=loss_object)

    # TODO: implement model.fit()

    logger.info('Training completed.')