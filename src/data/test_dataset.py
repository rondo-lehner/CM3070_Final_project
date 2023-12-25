# -*- coding: utf-8 -*-
import click
import logging
import datasets.deep_globe_2018
import tensorflow_datasets as tfds

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading deep globe dataset')

    ds, ds_info = tfds.load(
        # Useful link: https://github.com/tensorflow/datasets/issues/2680#issuecomment-778801923
        name='deep_globe_2018',
        download=False,
        with_info=True
    )
    logger.info(ds_info)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
