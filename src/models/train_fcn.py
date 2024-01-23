## Imports
import tensorflow as tf

import src.data.pipelines.fcn_pipeline as fcn_pipeline
import os
import gc
import datetime
import click
import logging

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models import fcn

## Pipeline
SPLIT_TRAIN = ":10"
SPLIT_VALID = "10:20"
SPLIT_TEST = "20:30"
BATCH_SIZE = 1
IMAGE_SIZE = 224

## Training
EPOCHS = 2
VAL_SUBSPLITS = 2
VALIDATION_STEPS = 10//BATCH_SIZE//VAL_SUBSPLITS
STEPS_PER_EPOCH = 10 // BATCH_SIZE

## Tensorboard
LOG_DIR = os.path.join(os.getcwd(), 'models', 'logs', 'fcn_32s', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
UPDATE_FREQ = 1

def main():
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')

    pipeline = fcn_pipeline.FcnPipeline(
        SPLIT_TRAIN,
        SPLIT_VALID,
        SPLIT_TEST,
        BATCH_SIZE,
        IMAGE_SIZE
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            update_freq=UPDATE_FREQ,
            write_images=False # True doesn't add real benefit, it appears there is a limitation with the visualisation of Conv2D weights: https://github.com/tensorflow/tensorboard/issues/2240
        )

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [loss,
            tf.keras.metrics.MeanIoU(num_classes=7)]

    fcn_32s = fcn.get_fcn_32s()
    fcn_32s.compile(optimizer=opt, loss=loss, metrics=metrics)

    fcn_32s.fit(pipeline.train, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=pipeline.valid,
                            callbacks=[tensorboard_callback])

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
