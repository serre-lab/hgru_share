import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


try:
    @tf.RegisterGradient('SymmetricConv')
    def _Conv2DGrad(op, grad):
        """Weight sharing for symmetric lateral connections."""
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        use_cudnn_on_gpu = op.get_attr('use_cudnn_on_gpu')
        data_format = op.get_attr('data_format')
        shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
        dx = nn_ops.conv2d_backprop_input(
            shape_0,
            op.inputs[1],
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = nn_ops.conv2d_backprop_filter(
            op.inputs[0],
            shape_1,
            grad,
            strides=strides,
            padding=padding,
            use_cudnn_on_gpu=use_cudnn_on_gpu,
            data_format=data_format)
        dw = 0.5 * (dw + tf.transpose(dw, (0, 1, 3, 2)))
        return dx, dw
except Exception, e:
    print str(e)
    print 'Already imported SymmetricConv.'
