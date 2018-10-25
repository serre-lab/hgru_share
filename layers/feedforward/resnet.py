# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks
(also known as ResNet v2).

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class model:
    """Base class for building the Resnet v2 Model.
    """

    def __init__(
            self,
            trainable=True,
            num_classes=1000,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            resnet_size=18,
            block_strides=[1, 2, 2, 2],
            data_format=None,
            apply_to='input',
            extra_convs=1,
            squash=tf.sigmoid,
            output_layer='final_dense',
            human_score_layer='final_avg_pool',
            probability_layer='prob'):
        """Creates a model for classifying an image.

        Args:
            resnet_size: A single integer for the size of the ResNet model.
            num_classes: The number of classes used as labels.
            num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent
                block layer.
            kernel_size: The kernel size to use for convolution.
            conv_stride: stride size for the initial convolutional layer
            first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer. Not
                used if first_pool_size is None.
            second_pool_size: Pool size to be used for the second pooling
                layer.
            second_pool_stride: stride size for the final pooling layer
            block_fn: Which block layer function should be used? Pass in one of
                the two functions defined above: building_block or
                bottleneck_block
            block_sizes: A list containing n values, where n is the number of
                sets of block layers desired. Each value should be the number
                of blocks in the i-th set.
            block_strides: List of integers representing the desired stride
                size for each of the sets of block layers. Should be same
                length as block_sizes.
            final_size: The expected size of the model after the second
                pooling.
            data_format: Input format ('channels_last', 'channels_first', or
                None). If set to None, the format is dependent on whether a GPU
                is available.
        """
        self.resnet_size = resnet_size
        block_sizes = _get_block_sizes(resnet_size)
        attention_blocks = _get_attention_sizes(resnet_size)
        data_format = 'channels_last'
        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            block_fn = self.building_block
            final_size = 512
        else:
            block_fn = self.bottleneck_block
            final_size = 2048

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_fn = block_fn
        self.block_sizes = block_sizes
        self.attention_blocks = attention_blocks
        self.block_strides = block_strides
        self.final_size = final_size
        self.output_layer = output_layer
        self.probability_layer = probability_layer
        self.apply_to = apply_to
        self.trainable = trainable
        self.squash = squash
        self.extra_convs = extra_convs
        self.attention_losses = []
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        if isinstance(self.squash, basestring):
            self.squash = interpret_nl(self.squash)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(
            self,
            rgb,
            dropout=0.5,
            training=True,
            feature_attention=False,
            return_up_to=None):

        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only
            when training the classifier.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        if return_up_to is not None:
            raise NotImplementedError('Need to return specific layers.')
        # rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8

        # Convert RGB to BGR
        # rgb_scaled = rgb_scaled - self.VGG_MEAN
        # assert rgb_scaled.get_shape().as_list()[1:] == [224, 224, 3]

        inputs = self.conv2d_fixed_padding(
            inputs=rgb,
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.conv_stride,
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        for i, (num_blocks, use_attention) in enumerate(
                zip(self.block_sizes, self.attention_blocks)):
            num_filters = self.num_filters * (2**i)
            if isinstance(use_attention, list):
                block_attention = [
                    feature_attention if x else '' for x in use_attention]
            else:
                block_attention = num_blocks * [
                    use_attention * feature_attention]
            assert num_blocks == len(
                block_attention), 'Fix your attention application.'
            inputs = self.block_layer(
                inputs=inputs,
                filters=num_filters,
                block_fn=self.block_fn,
                blocks=num_blocks,
                strides=self.block_strides[i],
                training=training,
                name='block_layer{}'.format(i + 1),
                data_format=self.data_format,
                feature_attention=block_attention)

        inputs = self.batch_norm_relu(inputs, training, self.data_format)
        inputs = tf.reduce_mean(inputs, reduction_indices=[1, 2])
        # inputs = tf.layers.average_pooling2d(
        #     inputs=inputs,
        #     pool_size=self.second_pool_size,
        #     strides=self.second_pool_stride,
        #     padding='VALID',
        #     data_format=self.data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        inputs = tf.reshape(inputs, [-1, self.final_size])
        inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
        final_dense = tf.identity(inputs, 'final_dense')
        prob = tf.nn.softmax(
            final_dense,
            name='softmax_tensor')
        setattr(self, self.output_layer, final_dense)
        setattr(self, self.probability_layer, prob)
        return inputs

    def feature_attention(
            self,
            bottom,
            global_pooling=tf.reduce_mean,
            intermediate_nl=tf.nn.leaky_relu,
            squash=tf.sigmoid,
            name=None,
            training=True,
            batchnorm=False,
            combine='sum_p',
            _BATCH_NORM_DECAY=0.997,
            _BATCH_NORM_EPSILON=1e-5,
            r=4,
            return_map=False):
        """https://arxiv.org/pdf/1709.01507.pdf"""
        # 1. Global pooling
        mu = global_pooling(
            bottom, reduction_indices=[1, 2], keep_dims=True)

        # 2. FC layer with c / r channels + a nonlinearity
        c = int(mu.get_shape()[-1])
        intermediate_size = int(c / r)
        intermediate_activities = intermediate_nl(
            self.fc_layer(
                bottom=tf.contrib.layers.flatten(mu),
                out_size=intermediate_size,
                name='%s_ATTENTION_intermediate' % name,
                training=training))

        # 3. FC layer with c / r channels + a nonlinearity
        out_size = c
        output_activities = self.fc_layer(
            bottom=intermediate_activities,
            out_size=out_size,
            name='%s_ATTENTION_output' % name,
            training=training)
        if squash is not None:
            output_activities = self.fc_layer(
                bottom=intermediate_activities,
                out_size=out_size,
                name='%s_ATTENTION_output' % name,
                training=training)

        # 4. Add batchnorm to scaled activities
        if batchnorm:
            bottom = tf.layers.batch_normalization(
                inputs=bottom,
                axis=3,
                momentum=_BATCH_NORM_DECAY,
                epsilon=_BATCH_NORM_EPSILON,
                center=True,
                scale=True,
                training=training,
                fused=True)

        # 5. Scale bottom with output_activities
        exp_activities = tf.expand_dims(
            tf.expand_dims(output_activities, 1), 1)
        if return_map:
            return exp_activities
        scaled_bottom = bottom * exp_activities

        # 6. Add a loss term to compare scaled activity to clickmaps
        if combine == 'sum_abs':
            salience_bottom = tf.reduce_sum(
                tf.abs(
                    scaled_bottom), axis=-1, keep_dims=True)
        elif combine == 'sum_p':
            salience_bottom = tf.reduce_sum(
                tf.pow(
                    scaled_bottom, 2), axis=-1, keep_dims=True)
        else:
            raise NotImplementedError(
                '%s combine not implmented.' % combine)
        self.attention_losses += [salience_bottom]
        return scaled_bottom

    def feature_attention_fc(
            self,
            bottom,
            intermediate_nl=tf.nn.relu,
            squash=tf.sigmoid,
            name=None,
            training=True,
            batchnorm=False,
            extra_convs=1,
            extra_conv_size=5,
            dilation_rate=(1, 1),
            intermediate_kernel=1,
            normalize_output=False,
            include_fa=True,
            interaction='both',  # 'additive',  # 'both',
            r=4):
        """Fully convolutional form of https://arxiv.org/pdf/1709.01507.pdf"""

        # 1. FC layer with c / r channels + a nonlinearity
        c = int(bottom.get_shape()[-1])
        intermediate_channels = int(c / r)
        intermediate_activities = tf.layers.conv2d(
            inputs=bottom,
            filters=intermediate_channels,
            kernel_size=intermediate_kernel,
            activation=intermediate_nl,
            padding='SAME',
            use_bias=True,
            kernel_initializer=tf.variance_scaling_initializer(),
            trainable=training,
            name='%s_ATTENTION_intermediate' % name)

        # 1a. Optionally add convolutions with spatial dimensions
        if extra_convs:
            for idx in range(extra_convs):
                intermediate_activities = tf.layers.conv2d(
                    inputs=intermediate_activities,
                    filters=intermediate_channels,
                    kernel_size=extra_conv_size,
                    activation=intermediate_nl,
                    padding='SAME',
                    use_bias=True,
                    dilation_rate=dilation_rate,
                    kernel_initializer=tf.variance_scaling_initializer(),
                    trainable=training,
                    name='%s_ATTENTION_intermediate_%s' % (name, idx))

        # 2. Spatial attention map
        output_activities = tf.layers.conv2d(
            inputs=intermediate_activities,
            filters=1,  # c,
            kernel_size=1,
            padding='SAME',
            use_bias=True,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(),
            trainable=training,
            name='%s_ATTENTION_output' % name)
        if batchnorm:
            output_activities = self.batch_norm_relu(
                inputs=output_activities,
                training=training,
                use_relu=False)

        # Also calculate se attention
        if include_fa:
            fa_map = self.feature_attention(
                bottom=bottom,
                intermediate_nl=intermediate_nl,
                squash=None,
                name=name,
                training=training,
                batchnorm=False,
                r=r,
                return_map=True)
            if interaction == 'both':
                k = fa_map.get_shape().as_list()[-1]
                alpha = tf.get_variable(
                    name='alpha_%s' % name,
                    shape=[1, 1, 1, k],
                    initializer=tf.variance_scaling_initializer())
                beta = tf.get_variable(
                    name='beta_%s' % name,
                    shape=[1, 1, 1, k],
                    initializer=tf.variance_scaling_initializer())
                additive = output_activities + fa_map
                multiplicative = output_activities * fa_map
                output_activities = alpha * additive + beta * multiplicative
                # output_activities = output_activities * fa_map
            elif interaction == 'multiplicative':
                output_activities = output_activities * fa_map
            elif interaction == 'additive':
                output_activities = output_activities + fa_map
            else:
                raise NotImplementedError(interaction)
        output_activities = squash(output_activities)

        # 3. Scale bottom with output_activities
        scaled_bottom = bottom * output_activities

        # 4. Use attention for a clickme loss
        if normalize_output:
            norm = tf.sqrt(
                tf.reduce_sum(
                    tf.pow(output_activities, 2),
                    axis=[1, 2],
                    keep_dims=True))
            self.attention_losses += [
                output_activities / (norm + 1e-12)]
        else:
            self.attention_losses += [output_activities]
        return scaled_bottom

    def batch_norm_relu(
            self,
            inputs,
            training,
            data_format=None,
            use_relu=True,
            _BATCH_NORM_DECAY=0.997,
            _BATCH_NORM_EPSILON=1e-5):
        """Performs a batch normalization followed by a ReLU."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=3,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
            center=True,
            scale=True,
            training=training,
            fused=True)
        inputs = tf.nn.relu(inputs)
        return inputs

    def fixed_padding(self, inputs, kernel_size, data_format):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on
                data_format.
            kernel_size: The kernel to be used in the conv2d or max_pool2d
            operation. Should be a positive integer.
            data_format: The input format ('channels_last' or
            'channels_first').

        Returns:
            A tensor with the same format as the input with the data either
            intact (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(
            inputs,
            [
                [0, 0],
                [pad_beg, pad_end],
                [pad_beg, pad_end],
                [0, 0]
            ])
        return padded_inputs

    def conv2d_fixed_padding(
            self,
            inputs,
            filters,
            kernel_size,
            strides,
            data_format):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not
        # on the dimensions of `inputs` (as opposed to using
        # `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def building_block(
            self,
            inputs,
            filters,
            training,
            projection_shortcut,
            strides,
            data_format,
            feature_attention=False,
            block_id=None):
        """Standard building block for residual networks with BN before convolutions.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in]
                or [batch, height_in, width_in, channels] depending on
                data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or
                inference mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will
                ultimately downsample the input.
            data_format: The input format ('channels_last' or
                'channels_first').

        Returns:
            The output tensor of the block.
        """
        shortcut = inputs
        inputs = self.batch_norm_relu(inputs, training, data_format)
        if self.apply_to == 'input':
            if feature_attention == 'paper':
                inputs = self.feature_attention(
                    bottom=inputs,
                    name=block_id,
                    training=training)
            elif feature_attention == 'fc':
                inputs = self.feature_attention_fc(
                    bottom=inputs,
                    name=block_id,
                    training=training,
                    squash=self.squash,
                    extra_convs=self.extra_convs)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)

        # Feature attention applied to the dense path
        if self.apply_to == 'output':
            if feature_attention == 'paper':
                inputs = self.feature_attention(
                    bottom=inputs,
                    name=block_id,
                    training=training)
            elif feature_attention == 'fc':
                inputs = self.feature_attention_fc(
                    bottom=inputs,
                    name=block_id,
                    training=training,
                    squash=self.squash,
                    extra_convs=self.extra_convs)

        return inputs + shortcut

    def bottleneck_block(
            self,
            inputs,
            filters,
            training,
            projection_shortcut,
            strides,
            data_format,
            feature_attention=False,
            block_id=None):
        """Bottleneck block variant for residual networks with BN before convolutions.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on
                data_format.
            filters: The number of filters for the first two convolutions. Note
                that the third and final convolution will use 4 times as many
                filters.
            training: A Boolean for whether the model is in training or
                inference mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will
                ultimately downsample the input.
            data_format: The input format ('channels_last' or
                'channels_first').

        Returns:
            The output tensor of the block.
        """
        shortcut = inputs
        inputs = self.batch_norm_relu(inputs, training, data_format)
        if self.apply_to == 'input':
            if feature_attention == 'paper':
                inputs = self.feature_attention(
                    bottom=inputs,
                    name=block_id,
                    training=training)
            elif feature_attention == 'fc':
                inputs = self.feature_attention_fc(
                    bottom=inputs,
                    name=block_id,
                    training=training,
                    squash=self.squash,
                    extra_convs=self.extra_convs)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)

        # Feature attention applied to the dense path
        if self.apply_to == 'output':
            if feature_attention == 'paper':
                inputs = self.feature_attention(
                    bottom=inputs,
                    name=block_id,
                    training=training)
            elif feature_attention == 'fc':
                inputs = self.feature_attention_fc(
                    bottom=inputs,
                    name=block_id,
                    training=training,
                    squash=self.squash,
                    extra_convs=self.extra_convs)
        return inputs + shortcut

    def block_layer(
            self,
            inputs,
            filters,
            block_fn,
            blocks,
            strides,
            training,
            name,
            data_format,
            feature_attention):
        """Creates one layer of blocks for the ResNet model.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on
                data_format.
            filters: The number of filters for the first two convolutions. Note
                that the third and final convolution will use 4 times as many
                filters.
            training: A Boolean for whether the model is in training or
                inference mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
                (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will
                ultimately downsample the input.
            data_format: The input format ('channels_last' or
                'channels_first').

        Returns:
            The output tensor of the block layer.
        """
        # Bottleneck blocks end with 4x # of filters as they start with
        filters_out = 4 * filters \
            if self.resnet_size >= 50 else filters

        def projection_shortcut(inputs):
            return self.conv2d_fixed_padding(
                inputs=inputs,
                filters=filters_out,
                kernel_size=1,
                strides=strides,
                data_format=data_format)

        # Only first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(
            inputs=inputs,
            filters=filters,
            training=training,
            projection_shortcut=projection_shortcut,
            strides=strides,
            data_format=data_format,
            feature_attention=feature_attention[0],
            block_id='%s_0' % name)

        for idx in range(1, blocks):
            inputs = block_fn(
                inputs=inputs,
                filters=filters,
                training=training,
                projection_shortcut=None,
                strides=1,
                data_format=data_format,
                feature_attention=feature_attention[idx],
                block_id='%s_%s' % (name, idx))
        return tf.identity(inputs, name)

    def conv_layer(
            self,
            bottom,
            in_channels=None,
            out_channels=None,
            name=None,
            training=True,
            stride=1,
            filter_size=3):
        """Method for creating a convolutional layer."""
        assert name is not None, 'Supply a name for your operation.'
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size=filter_size,
                in_channels=in_channels,
                out_channels=out_channels,
                name=name)
            conv = tf.nn.conv2d(
                bottom,
                filt,
                [1, stride, stride, 1],
                padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def get_conv_var(
            self,
            filter_size,
            in_channels,
            out_channels,
            name,
            init_type='scaling'):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return filters, biases

    def fc_layer(
            self,
            bottom,
            in_size=None,
            out_size=None,
            name=None,
            activation=True,
            dropout=None,
            training=True):
        """Method for creating a fully connected layer."""
        assert name is not None, 'Supply a name for your operation.'
        if in_size is None:
            in_size = int(bottom.get_shape()[-1])
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(
            self,
            in_size,
            out_size,
            name,
            init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self,
            initial_value,
            name,
            idx,
            var_name,
            in_size=None,
            out_size=None):
        value = initial_value

        if type(value) is list:
            var = tf.get_variable(
                name=var_name,
                shape=value[0],
                initializer=value[1],
                trainable=self.trainable)
        else:
            var = tf.get_variable(
                name=var_name,
                initializer=value,
                trainable=self.trainable)
        # self.var_dict[(name, idx)] = var
        return var


def _get_block_sizes(resnet_size):
    """The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = (
            'Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
    raise ValueError(err)


def _get_attention_sizes(resnet_size):
    """The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    """
    choices = {
        18: [False, [False, True], [True, False], False],
        34: [False, [False, True, True, False], False, False],
        50: [False, [False, False, False, False], True, False],
        101: [False, False, True, True],
        152: [False, False, True, True],
        200: [False, False, True, True]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = (
            'Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, choices.keys()))
    raise ValueError(err)


def interpret_nl(nl_string):
    """Return the tensorflow nonlinearity referenced in nl_string."""
    if nl_string == 'relu':
        return tf.nn.relu
    elif nl_string == 'leaky_relu':
        return tf.nn.leaky_relu
    elif nl_string == 'sigmoid':
        return tf.sigmoid
    elif nl_string == 'tanh':
        return tf.tanh
    else:
        raise NotImplementedError(nl_string)
