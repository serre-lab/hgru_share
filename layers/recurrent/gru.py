"""Contextual model with partial filters."""
import numpy as np
import tensorflow as tf
from ops import initialization


# Dependency for symmetric weight ops is in models/layers/ff.py
class GRU(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            layer_name,
            x_shape,
            timesteps=1,
            h_ext=15,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None,
            train=True):
        """Global initializations and settings."""
        self.n, self.h, self.w, self.k = x_shape
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding
        self.train = train
        self.layer_name = layer_name

        # Sort through and assign the auxilliary variables
        default_vars = self.defaults()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                default_vars[k] = v
        self.update_params(default_vars)

        # Kernel shapes
        self.h_ext = 2 * np.floor(h_ext / 2.0).astype(np.int) + 1
        self.h_shape = [self.h_ext, self.h_ext, self.k, self.k]
        self.g_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.m_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.bias_shape = [1, 1, 1, self.k]

        # Nonlinearities and initializations
        if isinstance(self.recurrent_nl, basestring):
            self.recurrent_nl = self.interpret_nl(self.recurrent_nl)

    def defaults(self):
        """A dictionary containing defaults for auxilliary variables.

        These are adjusted by a passed aux dict variable."""
        return {
            'dtype': tf.float32,
            'hidden_init': 'random',
            'gate_bias_init': 'chronos',
            'train': True,
            'recurrent_nl': tf.nn.tanh,
            'gate_nl': tf.nn.sigmoid,
            'normal_initializer': True,
            'gate_filter': 1,  # Gate kernel size
            'nu': True,  # subtractive eCRF
        }

    def interpret_nl(self, nl_type):
        """Return activation function."""
        if nl_type == 'tanh':
            return tf.nn.tanh
        elif nl_type == 'relu':
            return tf.nn.relu
        elif nl_type == 'selu':
            return tf.nn.selu
        elif nl_type == 'leaky_relu':
            return tf.nn.leaky_relu
        elif nl_type == 'hard_tanh':
            return lambda x: tf.maximum(tf.minimum(x, 1), 0)
        else:
            raise NotImplementedError(nl_type)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices.
        (np.prod([h, w, k]) / 2) - k params in the surround filter
        """
        with tf.variable_scope('%s_hgru_weights' % self.layer_name):
            self.horizontal_kernels = tf.get_variable(
                name='%s_horizontal' % self.layer_name,
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.h_shape,
                    uniform=self.normal_initializer),
                trainable=True)
            self.h_bias = tf.get_variable(
                name='%s_h_bias' % self.layer_name,
                initializer=initialization.xavier_initializer(
                    shape=self.bias_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.gain_kernels = tf.get_variable(
                name='%s_gain' % self.layer_name,
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.g_shape,
                    uniform=self.normal_initializer,
                    mask=None))
            self.mix_kernels = tf.get_variable(
                name='%s_mix' % self.layer_name,
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.m_shape,
                    uniform=self.normal_initializer,
                    mask=None))

            # Gain bias
            if self.gate_bias_init == 'chronos':
                bias_init = -tf.log(
                    tf.random_uniform(
                        self.bias_shape, minval=1, maxval=self.timesteps - 1))
            else:
                bias_init = tf.ones(self.bias_shape)
            self.gain_bias = tf.get_variable(
                name='%s_gain_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=True,
                initializer=bias_init)
            if self.gate_bias_init == 'chronos':
                bias_init = -bias_init
            else:
                bias_init = tf.ones(self.bias_shape)
            self.mix_bias = tf.get_variable(
                name='%s_mix_bias' % self.layer_name,
                dtype=self.dtype,
                trainable=True,
                initializer=bias_init)

    def conv_2d_op(
            self,
            data,
            weights):
        """2D convolutions for hgru."""
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            activities = tf.nn.conv2d(
                data,
                weights,
                self.strides,
                padding=self.padding)
        else:
            raise RuntimeError
        return activities

    def full(self, i0, x, h):
        """GRU body."""
        g1_intermediate = self.conv_2d_op(
            data=h,
            weights=self.gain_kernels)
        g1 = self.gate_nl(g1_intermediate + self.gain_bias)
        g2_intermediate = self.conv_2d_op(
            data=h,
            weights=self.mix_kernels)
        g2 = self.gate_nl(g2_intermediate + self.mix_bias)

        # Horizontal activities
        c = self.conv_2d_op(
            data=h * g1,
            weights=self.horizontal_kernels)
        h_tilda = self.recurrent_nl(x + c + self.h_bias)
        h = ((1 - g2) * h) + (g2 * h_tilda)

        # Interate loop
        i0 += 1
        return i0, x, h

    def condition(self, i0, x, h):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, x):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        if self.hidden_init == 'identity':
            h = tf.identity(x)
        elif self.hidden_init == 'random':
            h = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            h = tf.zeros_like(x)
        else:
            raise RuntimeError

        # While loop
        elems = [
            i0,
            x,
            h
        ]
        returned = tf.while_loop(
            self.condition,
            self.full,
            loop_vars=elems,
            back_prop=True,
            swap_memory=True)

        # Prepare output
        i0, x, h = returned
        return h
