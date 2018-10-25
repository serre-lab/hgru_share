import tensorflow as tf


def resize(x, size, method, align_corners=True, preserve_aspect_ratio=True):
    """Resize with method."""
    if method == 'bilinear':
        method = tf.image.ResizeMethod.BILINEAR
    elif method == 'bicubic':
        method = tf.image.ResizeMethod.BICUBIC
    elif method == 'nearest_neighbors' or method == 'nn':
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif method == 'area':
        method = tf.image.ResizeMethod.AREA
    return tf.image.resize_images(
        images=x,
        size=size,
        method=method,
        align_corners=align_corners,
        preserve_aspect_ratio=preserve_aspect_ratio)
