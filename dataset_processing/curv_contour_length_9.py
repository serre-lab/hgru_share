import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'curv_contour_length_9'
        self.contour_dir = '/media/data_cifs/curvy_2snakes_300/'
        self.im_extension = '.png'
        self.label_regex = r'(?<=length)\d+'
        self.config = Config()
        self.im_size = [300, 300]  # 600, 600
        self.model_input_image_size = [150, 150, 1]  # [107, 160, 3]
        self.max_ims = 125000
        self.output_size = [1]
        self.label_size = self.output_size
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.store_z = False
        self.normalize_im = False
        self.balance = True
        self.shuffle = True
        self.input_normalization = 'none'  # 'zscore'
        self.preprocess = ['']  # ['resize_nn']
        self.meta = os.path.join('metadata', 'combined.npy')
        self.negative = 'curv_contour_length_9_neg'
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.cv_split = 0.9
        self.cv_balance = True
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int64,
                'reshape': self.output_size
            }
        }

    def list_files(self, meta, directory):
        """List files from metadata."""
        files = []
        for f in meta:
            files += [
                os.path.join(
                    self.contour_dir,
                    directory,
                    f[0],
                    f[1])]
        return np.asarray(files)

    def get_data(self):
        """Get the names of files."""
        positive_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.name,
                self.meta))
        negative_meta = np.load(
            os.path.join(
                self.contour_dir,
                self.negative,
                self.meta))
        positive_ims = self.list_files(positive_meta, self.name)
        negative_ims = self.list_files(negative_meta, self.negative)
        positive_ims = positive_ims[
            np.random.permutation(len(positive_ims))]
        negative_ims = negative_ims[
            np.random.permutation(len(negative_ims))]
        if self.balance:
            pos_len = len(positive_ims)
            neg_len = len(negative_ims)
            min_len = np.min([pos_len, neg_len])
            if self.max_ims:
                min_len = np.min([min_len, self.max_ims])
            positive_ims = positive_ims[:min_len]
            negative_ims = negative_ims[:min_len]

        positive_labels = np.ones(len(positive_ims)).astype(int)
        negative_labels = np.zeros(len(negative_ims)).astype(int)
        all_ims = np.concatenate((positive_ims, negative_ims))
        all_labels = np.concatenate((positive_labels, negative_labels))
        num_ims = len(all_ims)
        if self.shuffle:
            rand_idx = np.random.permutation(num_ims)
            all_ims = all_ims[rand_idx]
            all_labels = all_labels[rand_idx]

        # Create CV folds
        cv_range = np.arange(num_ims)
        train_split = np.round(num_ims * self.cv_split)
        train_idx = cv_range < train_split
        valalidation_idx = cv_range >= train_split
        train_ims = all_ims[train_idx]
        valalidation_ims = all_ims[valalidation_idx]
        train_labels = all_labels[train_idx]
        validation_labels = all_labels[valalidation_idx]

        # Build CV dict
        cv_files, cv_labels = {}, {}
        cv_files[self.folds['train']] = train_ims
        cv_files[self.folds['val']] = valalidation_ims
        cv_labels[self.folds['train']] = train_labels
        cv_labels[self.folds['val']] = validation_labels
        return cv_files, cv_labels


