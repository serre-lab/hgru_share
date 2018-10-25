import os
import numpy as np
from glob import glob
from skimage import io
from tqdm import tqdm


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


def evaluate(
        config,
        sess,
        saver,
        val_dict,
        exp_label,
        images,
        im_ext,
        restore,
        smooth_its=50,
        stdev_spread=0.15,
        max_batches=1):
    """Run the model training loop."""
    assert max_batches > 0, 'Change max_batches to > 0.'
    files = glob(os.path.join(images, '*%s' % im_ext))
    num_batches = len(files) // config.batch_size
    num_batches = np.minimum(num_batches, max_batches).astype(int)
    e_grads, i_grads, og_ims, proc_ims, val_logits = [], [], [], [], []
    e_acts, i_acts = [], []
    saver.restore(sess, restore)
    for idx in range(num_batches):
        step = 0
        im_batch = []
        while step < config.batch_size:
            im_batch += [io.imread(files[step])]
            step += 1
        im_batch = np.stack(im_batch, axis=0)
        exc_grads, inh_grads = [], []
        for i in tqdm(range(smooth_its)):
            stdev = stdev_spread * (np.max(im_batch) - np.min(im_batch))
            noise_batch = np.copy(im_batch) + np.random.normal(
                0, stdev, im_batch.shape)
            feed_dict = {val_dict['val_images']: noise_batch}
            it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
            exc_grads += [it_val_dict['exc_val_gradients']]
            inh_grads += [it_val_dict['inh_val_gradients']]
        exc_grads = np.asarray(exc_grads).mean(0).squeeze()
        inh_grads = np.asarray(inh_grads).mean(0).squeeze()
        feed_dict = {val_dict['val_images']: im_batch}
        it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
        e_grads += [exc_grads]
        i_grads += [inh_grads]
        og_ims += [im_batch]
        proc_ims += [it_val_dict['proc_val_images']]
        val_logits += [it_val_dict['val_logits']]
        e_acts += [it_val_dict['exc_val_activities']]
        i_acts += [it_val_dict['inh_val_activities']]

    # Package output variables into a dictionary
    output_dict = {
        'e_grads': e_grads,
        'i_grads': i_grads,
        'e_acts': e_acts,
        'i_acts': i_acts,
        'og_ims': og_ims,
        'proc_ims': proc_ims,
        'val_logits': val_logits,
        'files': files
    }
    out_path = os.path.join(
        'movies',
        '%s_val_gradients' % exp_label)
    np.savez(
        out_path,
        **output_dict)
    print 'Saved gradients to: %s' % out_path
