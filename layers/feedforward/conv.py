import numpy as np
import tensorflow as tf


def conv_layer(
        bottom,
        name,
        num_filters=None,
        kernel_size=None,
        stride=[1, 1, 1, 1],
        padding='SAME',
        trainable=True,
        use_bias=True,
        reuse=False,
        aux={}):
    """2D convolutional layer with pretrained weights."""
    in_ch = int(bottom.get_shape()[-1])
    if 'transpose_inds' in aux.keys():
        transpose_inds = aux['transpose_inds']
    else:
        transpose_inds = False
    if 'pretrained' in aux.keys():
        kernel_initializer = np.load(aux['pretrained']).item()
        key = aux['pretrained_key']
        if key == 'weights':
            key = kernel_initializer.keys()[0]
        kernel_initializer, preloaded_bias = kernel_initializer[key]
        if not len(preloaded_bias) and use_bias:
            bias = tf.get_variable(
                name='%s_conv_bias' % name,
                initializer=tf.zeros_initializer(),
                shape=[1, 1, 1, kernel_initializer.shape[-1]],
                trainable=trainable)
        if transpose_inds:
            kernel_initializer = kernel_initializer.transpose(transpose_inds)
        kernel_size = kernel_initializer.shape[0]
        pretrained = True
    else:
        assert num_filters is not None, 'Describe your filters'
        assert kernel_size is not None, 'Describe your kernel_size'
        if 'initializer' in aux.keys():
            kernel_initializer = aux['initializer']
        else:
            # kernel_initializer = tf.variance_scaling_initializer()
            kernel_initializer = [
                [kernel_size, kernel_size, in_ch, num_filters],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        pretrained = False
    if pretrained:
        filters = tf.get_variable(
            name='%s_pretrained' % name,
            initializer=kernel_initializer,
            trainable=trainable)
    else:
        filters = tf.get_variable(
            name='%s_initialized' % name,
            shape=kernel_initializer[0],
            initializer=kernel_initializer[1],
            trainable=trainable)
        if use_bias:
            bias = tf.get_variable(
                name='%s_bias' % name,
                initializer=tf.zeros([1, 1, 1, num_filters]),
                trainable=trainable)
    activity = tf.nn.conv2d(
        bottom,
        filters,
        strides=stride,
        padding='SAME')
    if use_bias:
        activity += bias
    if 'nonlinearity' in aux.keys():
        if aux['nonlinearity'] == 'square':
            activity = tf.pow(activity, 2)
        elif aux['nonlinearity'] == 'relu':
            activity = tf.nn.relu(activity)
        else:
            raise NotImplementedError(aux['nonlinearity'])
    return activity

