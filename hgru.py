#!/usr/bin/env python
import os
import tensorflow as tf
from layers.feedforward import conv
from layers.feedforward import normalization
from layers.feedforward import pooling
from layers.recurrent import hgru
from config import Config
from utils import py_utils
from ops import training
from ops import metrics
from ops import data_structure
from ops import data_loader
from ops import optimizers
from ops import gradients


def experiment_params():
    """Parameters for the experiment."""
    exp = {
        'lr': [1e-3],
        'loss_function': ['cce'],
        'optimizer': ['nadam'],
        'dataset': [
            # 'curv_contour_length_9',
            'curv_contour_length_14',
            # 'curv_baseline',
        ]
    }
    exp['data_augmentations'] = [
        [
            'grayscale',
            'left_right',
            'up_down',
            'uint8_rescale',
            'singleton',
            'resize',
            # 'per_image_standardization',
            'zero_one'
        ]]
    exp['val_augmentations'] = exp['data_augmentations']
    exp['batch_size'] = 32  # Train/val batch size.
    exp['epochs'] = 16
    exp['exp_name'] = 'hgru_pathfinder_14'
    exp['model_name'] = 'hgru'
    # exp['clip_gradients'] = 7.
    exp['save_weights'] = True
    exp['validation_iters'] = 1000
    exp['num_validation_evals'] = 50
    exp['shuffle_val'] = True  # Shuffle val data.
    exp['shuffle_train'] = True
    return exp


def build_model(data_tensor, reuse, training):
    """Create the hgru from Learning long-range..."""
    with tf.variable_scope('cnn', reuse=reuse):
        with tf.variable_scope('input', reuse=reuse):
            conv_aux = {
                'pretrained': os.path.join(
                    'weights',
                    'gabors_for_contours_7.npy'),
                'pretrained_key': 's1',
                'nonlinearity': 'square'
            }
            x = conv.conv_layer(
                bottom=data_tensor,
                name='gabor_input',
                stride=[1, 1, 1, 1],
                padding='SAME',
                trainable=training,
                use_bias=True,
                aux=conv_aux)
            layer_hgru = hgru.hGRU(
                'hgru_1',
                x_shape=x.get_shape().as_list(),
                timesteps=8,
                h_ext=15,
                strides=[1, 1, 1, 1],
                padding='SAME',
                # aux={'gamma': False},
                train=training)
            h2 = layer_hgru.build(x)
            nh2 = normalization.batch(
                bottom=h2,
                name='hgru_bn',
                training=training)

        with tf.variable_scope('readout_1', reuse=reuse):
            activity = conv.conv_layer(
                bottom=nh2,
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
    return activity, h2


def main(
        experiment_name='hgru',
        gpu_device='/gpu:0'):
    """Run an experiment with hGRUs."""

    # Prepare to run the model
    config = Config()
    params = experiment_params()
    config = py_utils.add_to_config(
        d=params,
        config=config)
    exp_label = '%s_%s' % (experiment_name, py_utils.get_dt_stamp())
    summary_dir = os.path.join(config.summaries, exp_label)
    checkpoint_dir = os.path.join(config.checkpoints, exp_label)
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=config.dataset)
    dataset_module = dataset_module.data_processing()  # hardcoded class name
    train_key = [k for k in dataset_module.folds.keys() if 'train' in k]
    if not len(train_key):
        train_key = 'train'
    else:
        train_key = train_key[0]
    (
        train_data,
        train_means_image,
        train_means_label) = py_utils.get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=train_key)
    val_key = [k for k in dataset_module.folds.keys() if 'val' in k]
    if not len(val_key):
        val_key = 'train'
    else:
        val_key = val_key[0]
    val_data, val_means_image, val_means_label = py_utils.get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=val_key)

    # Create data tensors
    with tf.device('/cpu:0'):
        train_images, train_labels = data_loader.inputs(
            dataset=train_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.data_augmentations,
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle_train)  # ,
            # resize_output=config.resize_output)
        val_images, val_labels = data_loader.inputs(
            dataset=val_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.val_augmentations,
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle_val)  # ,
            # resize_output=config.resize_output)

    # Build training and val models
    with tf.device(gpu_device):
        train_logits, hgru_act = build_model(
            data_tensor=train_images,
            reuse=None,
            training=True)
        val_logits, _ = build_model(
            data_tensor=val_images,
            reuse=tf.AUTO_REUSE,
            training=False)

    # Derive loss
    train_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(train_labels, [-1]),
            logits=train_logits))
    val_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(val_labels, [-1]),
            logits=val_logits))
    train_accuracy = metrics.class_accuracy(
        labels=train_labels,
        logits=train_logits)
    val_accuracy = metrics.class_accuracy(
        labels=val_labels,
        logits=val_logits)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('val_accuracy', val_accuracy)
    tf.summary.image('train_images', train_images)
    tf.summary.image('val_images', val_images)

    # Build optimizer
    train_op = optimizers.get_optimizer(
        train_loss,
        config['lr'],
        config['optimizer'])

    # Initialize tf variables
    saver = tf.train.Saver(
        var_list=tf.global_variables())
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(
        summary_dir,
        sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create dictionaries of important training and validation information
    train_dict = {
        'train_loss': train_loss,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'hgru': hgru_act,
        'train_accuracy': train_accuracy
    }
    val_dict = {
        'val_loss': val_loss,
        'val_images': val_images,
        'val_labels': val_labels,
        'val_accuracy': val_accuracy,
    }

    # Create datastructure for saving data
    ds = data_structure.data(
        batch_size=config.batch_size,
        validation_iters=config.validation_iters,
        num_validation_evals=config.num_validation_evals,
        shuffle_val=config.shuffle_val,
        lr=config.lr,
        loss_function=config.loss_function,
        optimizer=config.optimizer,
        model_name=config.model_name,
        dataset=config.dataset,
        output_directory=config.results)

    # Start training loop
    training.training_loop(
        config=config,
        coord=coord,
        sess=sess,
        summary_op=summary_op,
        summary_writer=summary_writer,
        saver=saver,
        threads=threads,
        summary_dir=summary_dir,
        checkpoint_dir=checkpoint_dir,
        train_dict=train_dict,
        val_dict=val_dict,
        exp_label=exp_label,
        data_structure=ds)


if __name__ == '__main__':
    main()

