from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.callbacks import Callback

try:
    import requests
except ImportError:
    requests = None

class SubmissionGenerator(Callback):
    """Generates submission file after every epoch

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, period=1):
        super(ModelCheckpoint, self).__init__()

        self.period = period
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            print("Predicting labels on test set...")
            y_test = self.model.predict()
            with open("data/test_output.txt", "w") as f:
                f.write("Id,Prediction\n")
                for idx, y in zip(np.arange(y_test.shape[0]), y_test):
                    f.write(str(idx) + "," + str(y) + "\n")
