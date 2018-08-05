import tensorflow as tf


def batch(
        bottom,
        name,
        scale=True,
        center=True,
        renorm=True,
        training=True):
    return tf.layers.batch_normalization(
        inputs=bottom,
        name=name,
        scale=scale,
        center=center,
        renorm=renorm,
        training=training)

