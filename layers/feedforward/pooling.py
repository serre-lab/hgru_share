import numpy as np
import tensorflow as tf


def max_pool(
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME',
        data_format='NHWC'):
    """Local max pooling."""
    return tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        data_format=data_format,
        name=name)


def global_pool(
        bottom,
        name,
        aux,
        data_format='NHWC'):
    """Global avg pool. TODO: move to pool class."""
    if 'reduction_indices' in aux:
        reduction_indices = aux['reduction_indices']
    else:
        if data_format == 'NHWC':
            reduction_indices = [1, 2]
        elif data_format == 'NCHW':
            reduction_indices = [2, 3]
    if 'pool_type' in aux:
        pool_type = aux['pool_type']
    else:
        pool_type = 'max'
    if pool_type == 'average' or pool_type == 'mean':
        activity = tf.reduce_mean(
            bottom,
            reduction_indices=reduction_indices)
    elif pool_type == 'max':
        activity = tf.reduce_max(
            bottom,
            reduction_indices=reduction_indices)
    else:
        raise NotImplementedError('Cannot understand %s' % pool_type)
    return activity


def unpool_with_argmax_layer(
        bottom,
        ind,
        name,
        filter_size):
    """Pool/unpool operations."""
    if len(bottom.get_shape()) < 3:
        bottom = tf.expand_dims(bottom, axis=-1)
    input_shape = bottom.get_shape().as_list()
    output_shape = (
        input_shape[0],
        input_shape[1] * filter_size[0],
        input_shape[2] * filter_size[1],
        input_shape[3])
    target_shape = ind.get_shape().as_list()
    if np.any(input_shape != target_shape):
        bottom = tf.image.resize_image_with_crop_or_pad(
            bottom,
            target_shape[1],
            target_shape[2])
        input_shape = bottom.get_shape().as_list()
    flattened_input_size = np.prod(input_shape)
    flat_output_shape = [
        output_shape[0],
        output_shape[1] * output_shape[2] * output_shape[3]]
    bottom_ = tf.reshape(bottom, [flattened_input_size])
    batch_range = tf.reshape(
        tf.range(
            output_shape[0],
            dtype=ind.dtype),
        shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, [flattened_input_size, 1])
    ind_ = tf.reshape(ind, [flattened_input_size, 1])
    ind_ = tf.concat([b, ind_], 1)
    act = tf.scatter_nd(ind_, bottom_, shape=flat_output_shape)
    act = tf.reshape(act, output_shape)
    return act
