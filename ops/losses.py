import tensorflow as tf


def derive_loss(labels, logits, loss_type):
    """Derive loss_type between labels and logits."""
    if loss_type is None:
        loss_type = 'sparse_ce'

    if loss_type == 'sparse_ce':
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=logits))
    elif loss_type == 'sparse_ce_image':
        label_shape = labels.get_shape().as_list()
        if label_shape[-1] > 1:
            raise RuntimeError('Label shape is %s.' % label_shape)
        labels = tf.squeeze(labels)
        labels = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'bce_image':
        label_shape = labels.get_shape().as_list()
        if label_shape[-1] > 1:
            raise RuntimeError('Label shape is %s.' % label_shape)
        labels = tf.squeeze(labels)
        labels = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits))
    elif loss_type == 'l2':
        return tf.nn.l2_loss(tf.reshape(labels, [-1]) - logits)
    elif loss_type == 'l2_image':
        return tf.nn.l2_loss(labels - logits)
    elif loss_type == 'pearson' or loss_type == 'correlation':
        return pearson_dissimilarity(
            labels=labels,
            logits=logits,
            REDUCE=tf.reduce_mean)
    else:
        raise NotImplementedError(loss_type)


def pearson_dissimilarity(labels, logits, REDUCE, eps_1=1e-4, eps_2=1e-12):
    """Calculate pearson diss. loss."""
    pred = logits
    x_shape = pred.get_shape().as_list()
    y_shape = labels.get_shape().as_list()
    if x_shape[-1] == 1 and len(x_shape) == 2:
        # If calculating score across exemplars
        pred = tf.squeeze(pred)
        x_shape = [x_shape[0]]
        labels = tf.squeeze(labels)
        y_shape = [y_shape[0]]

    if len(x_shape) > 2:
        # Reshape tensors
        x1_flat = tf.contrib.layers.flatten(pred)
    else:
        # Squeeze off singletons to make x1/x2 consistent
        x1_flat = tf.squeeze(pred)
    if len(y_shape) > 2:
        x2_flat = tf.contrib.layers.flatten(labels)
    else:
        x2_flat = tf.squeeze(labels)
    x1_mean = tf.reduce_mean(x1_flat, keep_dims=True, axis=[-1]) + eps_1
    x2_mean = tf.reduce_mean(x2_flat, keep_dims=True, axis=[-1]) + eps_1

    x1_flat_normed = x1_flat - x1_mean
    x2_flat_normed = x2_flat - x2_mean

    count = int(x2_flat.get_shape()[-1])
    cov = tf.div(
        tf.reduce_sum(
            tf.multiply(
                x1_flat_normed, x2_flat_normed),
            -1),
        count)
    x1_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x1_flat - x1_mean),
                -1),
            count))
    x2_std = tf.sqrt(
        tf.div(
            tf.reduce_sum(
                tf.square(x2_flat - x2_mean),
                -1),
            count))
    corr = cov / (tf.multiply(x1_std, x2_std) + eps_2)
    if REDUCE is not None:
        corr = REDUCE(corr)
    return 1 - corr
