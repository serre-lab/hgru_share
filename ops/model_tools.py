import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import training
from ops import metrics
from ops import data_structure
from ops import data_loader
from ops import optimizers
from ops import losses
from ops import gradients


def get_placeholders(dataset_module, config):
    """Create placeholders and apply augmentations."""
    raise NotImplementedError
    train_images = tf.placeholder(
        dtype=dataset_module.tf_reader['image']['dtype'],
        shape=[config.batch_size] + dataset_module.im_size,
        name='train_images')
    train_labels = tf.placeholder(
        dtype=dataset_module.tf_reader['label']['dtype'],
        shape=[config.batch_size] + dataset_module.label_size,
        name='train_labels')
    val_images = tf.placeholder(
        dtype=dataset_module.tf_reader['image']['dtype'],
        shape=[config.batch_size] + dataset_module.im_size,
        name='val_images')
    val_labels = tf.placeholder(
        dtype=dataset_module.tf_reader['label']['dtype'],
        shape=[config.batch_size] + dataset_module.label_size,
        name='val_labels')
    aug_train_ims = []
    aug_train_labels = []
    aug_val_ims = []
    aug_val_labels = []
    split_train_ims = tf.split(train_images, config.batch_size, axis=0)
    split_train_labels = tf.split(train_labels, config.batch_size, axis=0)
    split_val_ims = tf.split(val_images, config.batch_size, axis=0)
    split_val_labels = tf.split(train_labels, config.batch_size, axis=0)
    for tr_im, tr_la, va_im, va_la in zip(
            split_train_ims,
            split_train_labels,
            split_val_ims,
            split_val_labels):
        tr_im, tr_la = data_loader.image_augmentations(
            image=tf.squeeze(tr_im),
            label=tf.squeeze(tr_la),
            model_input_image_size=dataset_module.model_input_image_size,
            data_augmentations=config.data_augmentations)
        va_im, va_la = data_loader.image_augmentations(
            image=tf.squeeze(va_im),
            label=tf.squeeze(va_la),
            model_input_image_size=dataset_module.model_input_image_size,
            data_augmentations=config.val_augmentations)
        aug_train_ims += [tr_im]
        aug_train_labels += [tr_la]
        aug_val_ims += [va_im]
        aug_val_labels += [va_la]
    aug_train_ims = tf.stack(aug_train_ims, axis=0)
    aug_train_labels = tf.stack(aug_train_labels, axis=0)
    aug_val_ims = tf.stack(aug_val_ims, axis=0)
    aug_val_labels = tf.stack(aug_val_labels, axis=0)
    return aug_train_ims, aug_train_labels, aug_val_ims, aug_val_labels


def model_builder(
        params,
        config,
        model_spec,
        gpu_device,
        cpu_device,
        placeholders=False,
        tensorboard_images=False):
    """Standard model building routines."""
    config = py_utils.add_to_config(
        d=params,
        config=config)
    exp_label = '%s_%s' % (params['exp_name'], py_utils.get_dt_stamp())
    directories = py_utils.prepare_directories(config, exp_label)
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
    if hasattr(config, 'val_dataset'):
        val_dataset = config.val_dataset
    else:
        val_dataset = config.dataset
    val_data, val_means_image, val_means_label = py_utils.get_data_pointers(
        dataset=val_dataset,
        base_dir=config.tf_records,
        cv=val_key)

    # Create data tensors

    with tf.device(cpu_device):
        if placeholders:
            (
                train_images,
                train_labels,
                val_images,
                val_labels) = get_placeholders(dataset_module, config)
            placeholders = dataset_module.get_data()
        else:
            train_images, train_labels = data_loader.inputs(
                dataset=train_data,
                batch_size=config.batch_size,
                model_input_image_size=dataset_module.model_input_image_size,
                tf_dict=dataset_module.tf_dict,
                data_augmentations=config.data_augmentations,
                num_epochs=config.epochs,
                tf_reader_settings=dataset_module.tf_reader,
                shuffle=config.shuffle_train)
            val_images, val_labels = data_loader.inputs(
                dataset=val_data,
                batch_size=config.batch_size,
                model_input_image_size=dataset_module.model_input_image_size,
                tf_dict=dataset_module.tf_dict,
                data_augmentations=config.val_augmentations,
                num_epochs=config.epochs,
                tf_reader_settings=dataset_module.tf_reader,
                shuffle=config.shuffle_val)

    # Build training and val models
    with tf.device(gpu_device):
        train_logits, train_hgru_act = model_spec(
            data_tensor=train_images,
            reuse=None,
            training=True)
        val_logits, val_hgru_act = model_spec(
            data_tensor=val_images,
            reuse=tf.AUTO_REUSE,
            training=False)

    # Derive loss
    loss_type = None
    if hasattr(config, 'loss_type'):
        loss_type = config.loss_type
    train_loss = losses.derive_loss(
        labels=train_labels,
        logits=train_logits,
        loss_type=loss_type)
    val_loss = losses.derive_loss(
        labels=val_labels,
        logits=val_logits,
        loss_type=loss_type)
    if hasattr(config, 'metric_type'):
        metric_type = config.metric_type
    else:
        metric_type = 'accuracy'
    if metric_type == 'pearson':
        train_accuracy = metrics.pearson_score(
            labels=train_labels,
            pred=train_logits,
            REDUCTION=tf.reduce_mean)
        val_accuracy = metrics.pearson_score(
            labels=val_labels,
            pred=val_logits,
            REDUCTION=tf.reduce_mean)
    else:
        train_accuracy = metrics.class_accuracy(
            labels=train_labels,
            logits=train_logits)
        val_accuracy = metrics.class_accuracy(
            labels=val_labels,
            logits=val_logits)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('val_accuracy', val_accuracy)
    if tensorboard_images:
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
        directories['summaries'],
        sess.graph)
    if not placeholders:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    else:
        coord, threads = None, None

    # Create dictionaries of important training and validation information
    train_dict = {
        'train_loss': train_loss,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'train_accuracy': train_accuracy
    }
    if isinstance(train_hgru_act, dict):
        for k, v in train_hgru_act.iteritems():
            train_dict[k] = v
    else:
        train_dict['activity'] = train_hgru_act
    val_dict = {
        'val_loss': val_loss,
        'val_images': val_images,
        'val_labels': val_labels,
        'val_accuracy': val_accuracy,
    }
    if isinstance(val_hgru_act, dict):
        for k, v in val_hgru_act.iteritems():
            val_dict[k] = v
    else:
        val_dict['activity'] = val_hgru_act

    # Count parameters
    num_params = np.sum(
        [np.prod(x.get_shape().as_list()) for x in tf.trainable_variables()])
    print 'Model has approximately %s trainable params.' % num_params

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
        num_params=num_params,
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
        directories=directories,
        train_dict=train_dict,
        val_dict=val_dict,
        exp_label=exp_label,
        data_structure=ds,
        placeholders=placeholders)
