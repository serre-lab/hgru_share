import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import logger


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def training_loop(
        config,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        summary_dir,
        checkpoint_dir,
        train_dict,
        val_dict,
        exp_label,
        data_structure,
        performance_metric='validation_loss',
        aggregator='max',
        save_checkpoints=False,
        save_gradients=False):
    """Run the model training loop."""
    log = logger.get(os.path.join(config.log_dir, exp_label))
    step = 0
    train_losses, train_accs, train_aux, timesteps = {}, {}, {}, {}
    val_losses, val_accs, val_scores, val_aux, val_labels = {}, {}, {}, {}, {}
    train_aux_check = np.any(['aux_score' in k for k in train_dict.keys()])
    val_aux_check = np.any(['aux_score' in k for k in val_dict.keys()])
    try:
        while not coord.should_stop():
            start_time = time.time()
            it_train_dict = sess.run(train_dict)
            duration = time.time() - start_time
            train_losses[step] = it_train_dict['train_loss']
            train_accs[step] = it_train_dict['train_accuracy']
            timesteps[step] = duration
            if train_aux_check:
                # Loop through to find aux scores
                it_train_aux = {
                    itk: itv
                    for itk, itv in it_train_dict.iteritems()
                    if 'aux_score' in itk}
                train_aux[step] = it_train_aux
            assert not np.isnan(
                it_train_dict['train_loss']).any(),\
                'Model diverged with loss = NaN'
            try:
                data_structure.update_training(
                    train_accuracy=train_accs[step],
                    train_loss=train_losses[step],
                    train_step=step)
                data_structure.save()
            except Exception as e:
                log.warning('Failed to update saver class: %s' % e)
            if step % config.validation_iters == 0:
                it_val_acc = np.asarray([])
                it_val_loss = np.asarray([])
                it_val_scores, it_val_labels, it_val_aux = [], [], []
                for num_vals in range(config.num_validation_evals):
                    # Validation accuracy as the average of n batches
                    it_val_dict = sess.run(val_dict)
                    it_val_acc = np.append(
                        it_val_acc,
                        it_val_dict['val_accuracy'])
                    it_val_loss = np.append(
                        it_val_loss,
                        it_val_dict['val_loss'])
                    it_val_labels += [it_val_dict['val_labels']]
                    it_val_scores += [it_val_dict['val_accuracy']]
                    if val_aux_check:
                        iva = {
                            itk: itv
                            for itk, itv in it_val_dict.iteritems()
                            if 'aux_score' in itk}
                        it_val_aux += [iva]
                val_acc = it_val_acc.mean()
                val_lo = it_val_loss.mean()
                val_accs[step] = val_acc
                val_losses[step] = val_lo
                val_scores[step] = it_val_scores
                val_labels[step] = it_val_labels
                val_aux[step] = it_val_aux
                try:
                    data_structure.update_validation(
                        validation_accuracy=val_accs[step],
                        validation_loss=val_losses[step],
                        validation_step=step)
                    data_structure.save()
                except Exception as e:
                    log.warning('Failed to update saver class: %s' % e)

                if save_checkpoints:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        'model_%s.ckpt' % step)
                    saver.save(
                        sess,
                        ckpt_path,
                        global_step=step)

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                log.info(format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    float(duration),
                    it_train_dict['train_accuracy'],
                    val_acc,
                    summary_dir))
            else:
                # Training status
                format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; '
                              # 'hgru mean %s | hgru std %s | '
                              '%.3f sec/batch) | Training accuracy = %s')
                log.info(format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    # it_train_dict['hgru'].mean(),
                    # it_train_dict['hgru'].std(),
                    float(duration),
                    it_train_dict['train_accuracy']))

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        log.info(
            'Done training for %d epochs, %d steps.' % (config.epochs, step))
        log.info('Saved to: %s' % checkpoint_dir)
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

    # Package output variables into a dictionary
    output_dict = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_aux': train_aux,
        'timesteps': timesteps,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_scores': val_scores,
        'val_labels': val_labels,
        'val_aux': val_aux,
    }
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
    return output_dict
