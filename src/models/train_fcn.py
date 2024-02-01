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
MODEL = 'fcn_16s'               # fcn_32s, fcn_16s, fcn_8s
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


    # TODO: Consider implementation of class weighted loss, a la: https://stackoverflow.com/a/69220169/6728108
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
   
    losses = {
        'train_loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
        'train_mIoU': tf.keras.metrics.MeanIoU(num_classes=7, name='train_mIoU', sparse_y_true=False, sparse_y_pred=False),
        'test_loss': tf.keras.metrics.Mean('test_loss', dtype=tf.float32),
        'test_mIoU': tf.keras.metrics.MeanIoU(num_classes=7, name='test_mIoU', sparse_y_true=False, sparse_y_pred=False)
    }

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(LOG_DIR, current_time, 'train')
    test_log_dir = os.path.join(LOG_DIR, current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    match MODEL:
        case 'fcn_32s':
            model = fcn.get_fcn_32s()
        case 'fcn_16s':
            fcn_32s_checkpoint_path = os.path.join(CHECKPOINT_DIR,'..', 'fcn_32s', 'val_loss: 1.371886968612671')
            model = fcn.get_fcn_16s(fcn_32s_checkpoint_path)

    step_global = 0
    if LOAD_WEIGHTS:
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        model.load_weights(latest)
    # fcn_32s.compile(optimizer=optimizer, loss=loss_object)

    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        step = 0
        p_bar = tf.keras.utils.Progbar(
            STEPS_PER_EPOCH,
            width=30,
            verbose=1,
            interval=0.05,
            stateful_metrics=['loss', 'mIoU', 'val_loss', 'val_mIoU'],
            unit_name='step'
        )

        # TODO: rework tensorboard logging to log per step and not per epoch
        for (x_train, y_train) in train:
            train_step(model, optimizer, loss_object, losses, x_train, y_train)
            p_bar.update(
                step,
                values=[
                    ('loss', losses['train_loss'].result()),
                    ('mIoU', losses['train_mIoU'].result()) 
                ]
            )
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', losses['train_loss'].result(), step=step_global)
                tf.summary.scalar('mIoU', losses['train_mIoU'].result(), step=step_global)
            step += 1
            step_global += 1
            

        for (x_test, y_test) in validation:
            test_step(model, loss_object, losses, x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', losses['test_loss'].result(), step=epoch)
            tf.summary.scalar('mIoU', losses['test_mIoU'].result(), step=epoch)

        p_bar.update(
            step,
            values=[
                ('loss', losses['train_loss'].result()),
                ('mIoU', losses['train_mIoU'].result()),
                ('val_loss', losses['test_loss'].result()),
                ('val_mIoU', losses['test_mIoU'].result())
            ],
            finalize=True
        )
        

        # template = 'Epoch {}, Loss: {}, mIoU: {}, Test Loss: {}, Test mIoU: {}'
        # print (template.format(epoch+1,
        #                         losses['train_loss'].result(), 
        #                         losses['train_mIoU'].result()*100,
        #                         losses['test_loss'].result(), 
        #                         losses['test_mIoU'].result()*100))
        if epoch % 2 == 0:
            model.save_weights(os.path.join(CHECKPOINT_DIR, f'{step_global}_val_loss:_{losses["test_loss"].result()}.h5'))

        # Reset metrics every epoch
        losses['train_loss'].reset_states()
        losses['train_mIoU'].reset_states()
        losses['test_loss'].reset_states()
        losses['test_mIoU'].reset_states()

    logger.info('Training completed.')

@tf.function
def train_step(model, optimizer, loss_object, losses, x_train, y_train):
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  losses['train_loss'](loss)
  losses['train_mIoU'](y_train, predictions)

@tf.function
def test_step(model, loss_object, losses, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  losses['test_loss'](loss)
  losses['test_mIoU'](y_test, predictions)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()