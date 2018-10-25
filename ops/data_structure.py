import os
import socket
import getpass
import numpy as np
from datetime import datetime


def make_dir(d):
    """Make directory if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


class data:
    """Class for packaging model results for the circuit nips 2018 paper.

    Params::

    validation_accuracy
        List of model validation accuracies. Each entry is a step of
        validation. We are evaluating this every 500 steps of training.
        Validation must be the average performance across 1000 randomly
        selected exemplars.
    train_accuracy
        List of model training accuracies. Each entry is a step of training.
    model_name
        String with an arbitrary model name.
    """
    def __init__(
            self,
            batch_size,
            validation_iters,
            num_validation_evals,
            shuffle_val,
            lr,
            loss_function,
            optimizer,
            model_name,
            dataset,
            num_params,
            output_directory):
        """Set model information as attributes."""
        self.create_meta()
        self.validation_loss = []
        self.validation_accuracy = []
        self.validation_step = []
        self.train_loss = []
        self.train_accuracy = []
        self.train_step = []
        self.batch_size = batch_size
        self.validation_iters = validation_iters
        self.num_validation_evals = num_validation_evals
        self.shuffle_val = shuffle_val
        self.lr = lr
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_name = model_name
        self.dataset = dataset
        self.num_params = num_params
        self.output_directory = output_directory
        if output_directory is not None:
            self.file_pointer = os.path.join(
                output_directory,
                '%s_%s' % (model_name, self.dt))
        else:
            self.file_pointer = os.path.join(
                '%s_%s' % (model_name, self.dt))
        make_dir(self.output_directory)
        self.validate()

    def required_keys(self):
        """Keys we need from data."""
        return [
            'validation_accuracy',
            'train_accuracy',
            'directory_pointer',
            'username',
            'homedir',
            'hostname',
            'dt'
        ]

    def create_meta(self):
        """Create meta information about this model."""
        # Get username
        self.username = getpass.getuser()
        self.homedir = os.environ['HOME']
        self.hostname = socket.gethostname()
        self.dt = str(
            datetime.now()).replace(
            ' ', '_').replace(
            ':', '_').replace('-', '_')

    def validate(self):
        """Validate that all information is included."""
        keys = self.required_keys()
        assert [k in keys for k in self.__dict__.keys()]
        assert isinstance(self.validation_accuracy, list),\
            'Pass a list of validation accuracies'
        assert isinstance(self.train_accuracy, list),\
            'Pass a list of training accuracies'

    def update_training(
            self, train_accuracy, train_loss, train_step):
        """Update training performance."""
        self.train_accuracy += [train_accuracy]
        self.train_loss += [train_loss]
        self.train_step += [train_step]

    def update_validation(
            self, validation_accuracy, validation_loss, validation_step):
        """Update validation performance."""
        self.validation_accuracy += [validation_accuracy]
        self.validation_loss += [validation_loss]
        self.validation_step += [validation_step]

    def save(self):
        """Save a npz with model info."""
        np.savez(
            self.file_pointer,
            **self.__dict__)
