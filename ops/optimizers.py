import tensorflow as tf


def get_optimizer(
        loss,
        lr,
        optimizer,
        model=None,
        clip_gradients=None,
        restore_scope=None,
        var_list=None,
        constraints=None):
    """Return the optimizer."""
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer == 'adam':
            optim = tf.train.AdamOptimizer
        elif optimizer == 'bsds_adam':
            optim = lambda x: tf.train.AdamOptimizer(x)
        elif optimizer == 'nadam':
            optim = tf.contrib.opt.NadamOptimizer
        elif optimizer == 'power':
            optim = tf.contrib.opt.PowerSignOptimizer
        elif optimizer == 'sgd':
            optim = tf.train.GradientDescentOptimizer
        elif optimizer == 'momentum':
            optim = momentum
        elif optimizer == 'rmsprop':
            optim = tf.train.RMSPropOptimizer
        else:
            raise RuntimeError('Cannot understand your loss function.')

        if optimizer == 'momentum':
            optim = optim(
                loss=loss,
                lr=lr,
                var_list=var_list,
                clip_gradients=clip_gradients)
        else:
            optim = optim(lr)
        if var_list:
            gvs = optim.compute_gradients(loss, var_list=var_list)
        else:
            gvs = optim.compute_gradients(loss)
        return check_and_clip_grads(gvs, optim, clip_gradients)


def check_and_clip_grads(
        gvs,
        optim,
        clip_gradients,
        visualize_gradients=False):
    """Check gradients for None and clip if requested."""
    null_grads = [x for x in gvs if x[0] is None]
    if len(null_grads):
        null_names = [x[1].name for x in null_grads]
        raise RuntimeError(
            'The following vars are not in the backprop graph: %s' %
            null_names)
    if clip_gradients:
        gradients, variables = zip(*gvs)
        # capped_grads, _ = tf.clip_by_global_norm(gradients, clip_gradients)
        capped_grads = []
        for v, grad in zip(gradients, variables):
            if 'context' in v.name:
                print '*' * 60
                print 'Clipping %s' % v.name
                print '*' * 60
                capped_grads += [tf.clip_by_norm(grad, clip_gradients)]
            else:
                capped_grads += [grad]
        if visualize_gradients:
            for g, v in zip(gradients, variables):
                tf.summary.histogram('grad_%s' % v.name, g)
        return optim.apply_gradients(
            zip(gradients, variables),
            global_step=tf.train.get_or_create_global_step())
    else:
        return optim.apply_gradients(gvs)


def momentum(loss, lr, var_list=None, momentum=0.9, clip_gradients=None):
    """Wrapper for SGD with momentum."""
    optim = tf.train.MomentumOptimizer(
        learning_rate=lr,
        momentum=momentum)
    return optim
