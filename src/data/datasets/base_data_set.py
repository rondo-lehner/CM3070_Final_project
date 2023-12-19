import numpy as np


class BaseDataSet:
    def __init__(self, config):
        self.config = config
        # load data here

    def next_batch(self, batch_size):
        pass