import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob


class data_processing(object):
    """Randomly select 10% of shapes for test. Only include lights 1/2/3."""
    def __init__(self):
        self.name = 'shapes'
        self.output_name = 'shapes_held_out_shape'
        self.image_dir = '/media/data_cifs/image_datasets/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d'
        self.config = Config()
        self.im_size = [500, 500]
        self.model_input_image_size = [250, 250, 1]  # [107, 160, 3]
        self.max_ims = None
        self.output_size = self.model_input_image_size
        self.label_size = self.output_size
        self.default_loss_function = 'l2'
        self.score_metric = 'pearson'
        self.store_z = False
        self.normalize_im = False
        self.shuffle = True
        self.img_file_ids = [
            'img1',
            'img2',
            'img3'
        ]
        self.label_string = 'depths'
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['']  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = False
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def shuffle_lists(self, files, labels):
        """Shuffle files and labels together."""
        num_files = len(files)
        rand_idx = np.random.permutation(num_files)
        files = files[rand_idx]
        labels = labels[rand_idx]
        return files, labels

    def get_data(self):
        """Get the names of files."""
        # (1) list all folders
        folders = glob(
            os.path.join(
                self.image_dir,
                self.name,
                'shape_*'))

        # (2) For each folder append _img*.png
        file_names = []
        label_names = []
        for f in folders:
            folder_name = f.split(os.path.sep)[-1]
            for idx in self.img_file_ids:
                file_names += [
                    os.path.join(
                        f,
                        '%s_%s%s' % (
                            folder_name,
                            idx,
                            self.im_extension))]
                label_names += [
                    os.path.join(
                        f,
                        '%s_%s%s' % (
                            folder_name,
                            self.label_string,
                            self.im_extension))]
        assert len(file_names), 'Could not find any files.'

        # (3) Split file list up appropriately
        file_names = np.asarray(file_names)
        label_names = np.asarray(label_names)
        file_names, label_names = self.shuffle_lists(
            files=file_names,
            labels=label_names)
        train_idx = int(len(file_names) * self.cv_split)
        train_files = file_names[:train_idx]
        train_labels = label_names[:train_idx]
        test_files = file_names[train_idx:]
        test_labels = label_names[train_idx:]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_files
        cv_files[self.folds['val']] = test_files
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = test_labels
        return cv_files, cv_labels
