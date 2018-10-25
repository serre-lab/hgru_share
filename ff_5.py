#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from config import Config
from ops import model_tools
from argparse import ArgumentParser


def experiment_params():
    """Parameters for the experiment."""
    exp = {}
    exp['lr'] = 1e-3
    exp['loss_function'] = 'cce'
    exp['optimizer'] = 'nadam'
    exp['dataset'] = 'curv_contour_length_14_full'
    exp['data_augmentations'] = [
        [
            'grayscale',
            # 'left_right',
            # 'up_down',
            'uint8_rescale',
            'singleton',
            'resize',
            # 'per_image_standardization',
            'zero_one'
        ]]
    exp['val_augmentations'] = exp['data_augmentations']
    exp['batch_size'] = 32  # Train/val batch size.
    exp['epochs'] = 2
    exp['model_name'] = __file__.split('.')[0]
    exp['exp_name'] = '%s_%s' % (
        exp['model_name'].split(os.path.sep)[-1],
        exp['dataset'])
    # exp['clip_gradients'] = 7.
    exp['save_weights'] = True
    exp['validation_iters'] = 1000
    exp['num_validation_evals'] = 50
    exp['shuffle_val'] = True  # Shuffle val data.
    exp['shuffle_train'] = True
    return exp


def build_model(data_tensor, reuse, training):
    """Create the gru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            conv_aux = {
                'pretrained': os.path.join(
                    'weights',
                    'gabors_for_contours_7.npy'),
                'pretrained_key': 's1',
                'nonlinearity': 'square'
            }
            num_filters = 9
            kernel_size = 20
            x = conv.conv_layer(
                bottom=data_tensor,
                name='gabor_input',
                stride=[1, 1, 1, 1],
                padding='SAME',
                trainable=training,
                use_bias=True,
                aux=conv_aux)
            activity = conv.conv_layer(
                bottom=x,
                name='c1',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c1_bn',
                training=training)
            activity = tf.nn.relu(activity)
            activity = conv.conv_layer(
                bottom=activity,
                name='c2',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c2_bn',
                training=training)
            activity = tf.nn.relu(activity)
            activity = conv.conv_layer(
                bottom=activity,
                name='c3',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c3_bn',
                training=training)
            activity = tf.nn.relu(activity)
            activity = conv.conv_layer(
                bottom=activity,
                name='c4',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c4_bn',
                training=training)
            activity = tf.nn.relu(activity)
            activity = conv.conv_layer(
                bottom=activity,
                name='c5',
                num_filters=num_filters,
                kernel_size=kernel_size,
                trainable=training,
                use_bias=False)
            activity = normalization.batch(
                bottom=activity,
                name='c5_bn',
                training=training)
            activity = tf.nn.relu(activity)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=activity,
                name='pre_readout_conv',
                num_filters=2,
                kernel_size=1,
                trainable=training,
                use_bias=False)
            pool_aux = {'pool_type': 'max'}
            activity = pooling.global_pool(
                bottom=activity,
                name='pre_readout_pool',
                aux=pool_aux)
            activity = normalization.batch(
                bottom=activity,
                name='readout_1_bn',
                training=training)

        with tf.variable_scope('readout_2', reuse=reuse):
            activity = tf.layers.flatten(
                activity,
                name='flat_readout')
            activity = tf.layers.dense(
                inputs=activity,
                units=2)
    return activity


def main(placeholders=False, gpu_device='/gpu:0', cpu_device='/cpu:0'):
    """Run an experiment with hGRUs."""
    config = Config()
    params = experiment_params()
    model_tools.model_builder(
        params=params,
        config=config,
        model_spec=build_model,
        placeholders=placeholders,
        gpu_device=gpu_device,
        cpu_device=cpu_device)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--placeholders',
        dest='placeholders',
        action='store_true',
        help='Use placeholder data loading.')
    args = parser.parse_args()
    main(**vars(args))
