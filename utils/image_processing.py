"""Useful functions for processing images."""
import cv2
import numpy as np


def crop_center(img, crop_size):
    """Center crop images."""
    im_shape = img.shape
    h, w = im_shape[:2]
    ch, cw = crop_size[:2]
    h_check = h <= ch
    w_check = w <= cw
    if h_check or w_check:
        return resize(img, crop_size)
    starth = h // 2 - (ch // 2)
    startw = w // 2 - (cw // 2)
    if len(im_shape) == 2:
        return img[starth:starth + ch, startw:startw + cw]
    elif len(im_shape) == 3:
        return img[starth:starth + ch, startw:startw + cw, :]
    else:
        raise NotImplementedError(
            'Cannot handle im size of %s' % len(im_shape))


def resize(img, new_size):
    """Resize image."""
    return cv2.resize(
        img,
        tuple(new_size[:2]),
        interpolation=cv2.INTER_CUBIC)


def pad_square(img):
    """Pad rectangular image to square."""
    im_shape = img.shape[:2]
    target_size = np.max(im_shape)
    h_pad = target_size - im_shape[0]
    w_pad = target_size - im_shape[1]
    t = h_pad // 2
    b = t
    l = w_pad // 2
    r = l
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, 0.)
