"""Model training with tfrecord queues or placeholders."""
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import logger
from utils import py_utils
from ops import data_to_tfrecords
from tqdm import tqdm


def training_step(
        sess,
        train_dict,
        feed_dict=False):
    """Run a step of training."""
    start_time = time.time()
    if feed_dict:
        it_train_dict = sess.run(train_dict, feed_dict=feed_dict)
    else:
        it_train_dict = sess.run(train_dict)
    train_acc_list = it_train_dict['train_accuracy']
    train_loss_list = it_train_dict['train_loss']
    duration = time.time() - start_time
    timesteps = duration
    return train_acc_list, train_loss_list, it_train_dict, timesteps


def validation_step(
        sess,
        val_dict,
        config,
        log,
        val_images=False,
        val_labels=False,
        val_batch_idx=False,
        val_batches=False):
    it_val_acc = np.asarray([])
    it_val_loss = np.asarray([])
    start_time = time.time()
    if val_batch_idx:
        shuff_val_batch_idx = val_batch_idx[
            np.random.permutation(len(val_batch_idx))]
    for num_vals in range(config.num_validation_evals):
        # Validation accuracy as the average of n batches
        if val_images:
            it_idx = shuff_val_batch_idx == num_vals
            it_ims = val_images[it_idx]
            it_labs = val_labels[it_idx]
            if isinstance(it_labs[0], basestring):
                it_labs = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_labs])
            feed_dict = {
                val_dict['val_images']: it_ims,
                val_dict['val_labels']: it_labs
            }
            it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
        else:
            it_val_dict = sess.run(val_dict)
        it_val_acc = np.append(
            it_val_acc,
            it_val_dict['val_accuracy'])
        it_val_loss = np.append(
            it_val_loss,
            it_val_dict['val_loss'])
    val_acc = it_val_acc.mean()
    val_lo = it_val_loss.mean()
    duration = time.time() - start_time
    return val_acc, val_lo, it_val_dict, duration


def save_progress(
        config,
        weight_dict,
        it_val_dict,
        exp_label,
        step,
        directories,
        sess,
        saver,
        data_structure,
        val_acc,
        val_lo,
        train_acc,
        train_loss,
        timesteps,
        log,
        summary_op,
        summary_writer,
        save_activities,
        save_checkpoints):
    """Save progress and important data."""
    if config.save_weights:
        it_weights = {
            k: it_val_dict[k] for k in weight_dict.keys()}
        py_utils.save_npys(
            data=it_weights,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])
    if save_activities:
        py_utils.save_npys(
            data=it_val_dict,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    if save_checkpoints:
        ckpt_path = os.path.join(
            directories['checkpoints'],
            'model_%s.ckpt' % step)
        saver.save(
            sess,
            ckpt_path,
            global_step=step)
    try:
        data_structure.update_validation(
            validation_accuracy=val_acc,
            validation_loss=val_lo,
            validation_step=step)
        data_structure.save()
    except Exception as e:
        log.warning('Failed to save validation info: %s' % e)
    try:
        data_structure.update_training(
            train_accuracy=train_acc,
            train_loss=train_loss,
            train_step=timesteps)
        data_structure.save()
    except Exception as e:
        log.warning('Failed to save training info: %s' % e)

    # Summaries
    try:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
    except Exception:
        print 'Failed to update summaries.'


def training_loop(
        config,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        directories,
        train_dict,
        val_dict,
        exp_label,
        data_structure,
        placeholders=False,
        save_checkpoints=False,
        save_activities=True,
        save_gradients=False):
    """Run the model training loop."""
    log = logger.get(os.path.join(config.log_dir, exp_label))
    step = 0
    if config.save_weights:
        try:
            weight_dict = {v.name: v for v in tf.trainable_variables()}
            val_dict = dict(
                val_dict,
                **weight_dict)
        except Exception:
            raise RuntimeError('Failed to find weights to save.')
    if placeholders:
        placeholder_images = placeholders[0]
        placeholder_labels = placeholders[1]
        train_images = placeholder_images['train']
        val_images = placeholder_images['val']
        train_labels = placeholder_labels['train']
        val_labels = placeholder_labels['val']
        train_batches = len(train_images) / config.batch_size
        train_batch_idx = np.arange(
            train_batches / config.batch_size).reshape(-1, 1).repeat(
                config.batch_size)
        train_images = train_images[:len(train_batch_idx)]
        train_labels = train_labels[:len(train_batch_idx)]
        val_batches = len(val_images) / config.batch_size
        val_batch_idx = np.arange(
            val_batches / config.batch_size).reshape(-1, 1).repeat(
                config.batch_size)
        val_images = val_images[:len(val_batch_idx)]
        val_labels = val_labels[:len(val_batch_idx)]
        for epoch in tqdm(
                range(config.epochs), desc='Epoch', total=config.epochs):
            for train_batch in range(train_batches):
                data_idx = train_batch_idx == train_batch
                it_train_images = train_images[data_idx]
                it_train_labels = train_labels[data_idx]
                if isinstance(it_train_images[0], basestring):
                    it_train_images = np.asarray(
                        [
                            data_to_tfrecords.load_image(im)
                            for im in it_train_images])
                feed_dict = {
                    train_dict['train_images']: it_train_images,
                    train_dict['train_labels']: it_train_labels
                }
                (
                    train_acc,
                    train_loss,
                    it_train_dict,
                    timesteps) = training_step(
                    sess=sess,
                    train_dict=train_dict,
                    feed_dict=feed_dict)
                if step % config.validation_iters == 0:
                    val_acc, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        data_structure=data_structure,
                        config=config,
                        log=log,
                        val_images=val_images,
                        val_labels=val_labels,
                        val_batch_idx=val_batch_idx,
                        val_batches=val_batches)

                    # Save progress and important data
                    save_progress(
                        config=config,
                        weight_dict=weight_dict,
                        it_val_dict=it_val_dict,
                        exp_label=exp_label,
                        step=step,
                        directories=directories,
                        sess=sess,
                        saver=saver,
                        data_structure=data_structure,
                        val_acc=val_acc,
                        val_lo=val_lo,
                        train_acc=train_acc,
                        train_loss=train_loss,
                        timesteps=timesteps,
                        log=log,
                        summary_op=summary_op,
                        summary_writer=summary_writer,
                        save_activities=save_activities,
                        save_checkpoints=save_checkpoints)
                    # Training status and validation accuracy
                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s | '
                        'Validation accuracy = %s | logdir = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        train_acc,
                        config.batch_size / duration,
                        float(duration),
                        train_loss,
                        val_acc,
                        directories['summaries']))
                else:
                    # Training status
                    format_str = (
                        '%s: step %d, loss = %.5f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        train_loss,
                        config.batch_size / duration,
                        float(duration),
                        train_acc))
                step += 1

    else:
        try:
            while not coord.should_stop():
                (
                    train_acc,
                    train_loss,
                    it_train_dict,
                    duration) = training_step(
                    sess=sess,
                    train_dict=train_dict)
                if step % config.validation_iters == 0:
                    val_acc, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log)

                    # Save progress and important data
                    save_progress(
                        config=config,
                        weight_dict=weight_dict,
                        it_val_dict=it_val_dict,
                        exp_label=exp_label,
                        step=step,
                        directories=directories,
                        sess=sess,
                        saver=saver,
                        data_structure=data_structure,
                        val_acc=val_acc,
                        val_lo=val_lo,
                        train_acc=train_acc,
                        train_loss=train_loss,
                        timesteps=duration,
                        log=log,
                        summary_op=summary_op,
                        summary_writer=summary_writer,
                        save_activities=save_activities,
                        save_checkpoints=save_checkpoints)

                    # Training status and validation accuracy
                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s | '
                        'Validation accuracy = %s | logdir = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        train_acc,
                        config.batch_size / duration,
                        float(duration),
                        train_loss,
                        val_acc,
                        directories['summaries']))
                else:
                    # Training status
                    format_str = (
                        '%s: step %d, loss = %.5f (%.1f examples/sec; '
                        '%.3f sec/batch) | Training accuracy = %s')
                    log.info(format_str % (
                        datetime.now(),
                        step,
                        train_loss,
                        config.batch_size / duration,
                        float(duration),
                        train_acc))

                # End iteration
                step += 1
        except tf.errors.OutOfRangeError:
            log.info(
                'Done training for %d epochs, %d steps.' % (
                    config.epochs, step))
            log.info('Saved to: %s' % directories['checkpoints'])
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    # Package output variables into a dictionary
    if save_gradients:
        np.savez(
            os.path.join(
                config.results,
                '%s_train_gradients' % exp_label),
            **it_train_dict)
        np.savez(
            os.path.join(
                config.results,
                '%s_val_gradients' % exp_label),
            **it_val_dict)
    return
