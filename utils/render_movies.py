import os
import numpy as np
import matplotlib
from utils import py_utils
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
plt.close('all')


def colorbar(mappable, max_val, min_val):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, ticks=[max_val, min_val])


def create_subplots(
        pos_path,
        neg_path,
        path_im,
        label,
        out_path,
        idx,
        show=True):
    """Create subplot figure and npzs."""
    fig = plt.figure(figsize=(14, 5))
    plt.suptitle(label)
    plt.xlabel('Timesteps')
    timesteps = len(pos_path)
    max_p = pos_path.max()
    max_n = neg_path.max()
    gs = gridspec.GridSpec(
        2,
        timesteps + 4,
        wspace=0.01,
        hspace=0.05,
        top=1. - 0.5 / (3),
        bottom=0.5 / (3),
        left=0.5 / (timesteps + 1), right=1 - 0.5 / (timesteps + 1))
    r, c = 0, 4
    for idx, (pp, ep) in enumerate(zip(pos_path, neg_path)):
        ax1 = plt.subplot(gs[r, c])
        img1 = ax1.imshow(pp, cmap='Reds', vmax=max_p, vmin=-0)
        ax1.axis('off')
        ax2 = plt.subplot(gs[r + 1, c])
        img2 = ax2.imshow(ep, cmap='Blues', vmax=max_n, vmin=-0)
        ax2.axis('off')
        if idx == (timesteps - 1):
            colorbar(img1, max_p, -0)
            colorbar(img2, max_n, -0)
            plt.tight_layout(h_pad=1)
        c += 1
    ax = plt.subplot(1, (timesteps + 4) // 2, 1)
    ax.imshow(path_im, cmap='Greys')
    ax.axis('off')
    ax = plt.subplot(1, (timesteps + 4) // 2, 6)
    ax.imshow(pos_path[-1] - neg_path[-1], cmap='RdBu_r')
    ax.axis('off')
    plt.savefig(
        os.path.join(
            out_path,
            '%s_%s_subplots.png' % (idx, label.replace(' ', '_'))))
    np.savez(
        os.path.join(
            out_path,
            '%s_%s' % (idx, label.replace(' ', '_'))),
        pos_path=pos_path, neg_path=neg_path, path_im=path_im)
    if show:
        plt.show()
    plt.close(fig)


def split_grad(grad, normalize=False):
    """Split gradient into +/- contributions."""
    pos = np.maximum(grad, 0)
    neg = np.abs(np.minimum(grad, 0))
    if normalize:
        pos /= normalize[0]
        neg /= normalize[1]
    return pos, neg


def interpolate(ims, interp_frames):
    """Interpolate ims to interp_frames length."""
    num_frames, xs, ys = ims.shape
    interp_array = np.zeros((interp_frames, xs, ys))
    for x in range(xs):
        for y in range(ys):
            interp_array[:, x, y] = np.interp(
                np.linspace(0, num_frames, interp_frames),
                range(num_frames),
                ims[:, x, y])
    return interp_array


def animate_frames(frames, cmap='Reds', vmax=1., vmin=0., fig=None):
    """Animate frames in matplotlib."""
    if fig is None:
        fig = plt.figure()
    im = plt.imshow(frames[0], cmap=cmap, vmax=vmax, vmin=vmin)

    def updatefig(j):
        """Update the animation."""
        im.set_array(frames[j])
        return [im]
    # kick off the animation
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(len(frames)),
        interval=50,
        blit=True)
    return fig, ani


def subplot_animate_frames(
        image,
        neg,
        pos,
        ag,
        label,
        im_cmap='Greys',
        neg_cmap='Blues',
        pos_cmap='Reds',
        ag_cmap='Greens'):
    # Setup figure and subplots
    font = {'size': 9}
    matplotlib.rc('font', **font)
    f0 = plt.figure(num=0, figsize=(12, 8))
    f0.suptitle(label, fontsize=12)
    # ax02 = plt.subplot2grid((2, 2), (0, 1))
    # ax03 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)
    # ax01 = plt.subplot2grid((2, 2), (1, 0))
    ax01 = plt.subplot(1, 4, 1)
    ax02 = plt.subplot(1, 4, 2)
    ax03 = plt.subplot(1, 4, 3)
    ax04 = plt.subplot(1, 4, 4)

    # Set titles of subplots
    ax01.set_title('Pathfinder image')
    ax02.set_title('Positive gradients')
    ax03.set_title('Negative gradients')
    ax04.set_title('Absolute gradients')

    # Turn off grids
    ax01.axis('off')
    ax02.axis('off')
    ax03.axis('off')
    ax04.axis('off')

    # set plots
    p11 = ax01.imshow(
        image.squeeze(), 
        cmap=im_cmap,
        label="Image")
    p12 = ax02.imshow(
        pos[0].squeeze(),
        cmap=pos_cmap,
        vmax=1.,
        vmin=0.,
        label="Positive gradient")
    p13 = ax03.imshow(
        neg[0].squeeze(),
        cmap=neg_cmap,
        vmax=1.,
        vmin=0.,
        label="Negative gradient")
    p14 = ax04.imshow(
        ag[0].squeeze(),
        cmap=ag_cmap,
        label="abs gradient")

    # Prepare update function
    def updateData(i, image, pos, neg, ag):
        # p11.set_array(image[i0])  # Don't update image
        # p11.set_array(image)
        print i
        p12.set_array(pos[i])
        p13.set_array(neg[i])
        p14.set_array(ag[i])
        return p11, p12, p13, p14

    ani = animation.FuncAnimation(
        f0,
        updateData,
        blit=False,
        frames=len(pos),
        fargs=(image, pos, neg, ag),
        interval=20,
        repeat=False)
    return f0, ani


def render_movie(
        images,
        grads,
        labels,
        idx,
        interp_frames,
        out_path,
        plot_movies=False,
        dpi=300):
    """Render a movie for an image/+ & - gradient across time."""
    # Select examples
    path_im = images[idx].squeeze()
    path_lab = labels[idx].squeeze()
    path_grad = grads[:, :, idx].squeeze()
    label = 'Postive path'
    if path_lab == 0:
        label = 'Negative path'
    # no_path_im = images[path_index].squeeze()
    # no_path_lab = labels[path_index].squeeze()
    # no_path_grad = grads[:, :, path_index].squeeze()

    # Split and normalize grads
    # max_pos = np.percentile(np.maximum(grads, 0), 95, axis=(1, 2, 3, 4, 5))
    # max_neg = np.percentile(np.abs(np.minimum(grads, 0)), 95, axis=(1, 2, 3, 4, 5))
    # max_pos = max_pos[:, None, None]
    # max_neg = max_neg[:, None, None]
    # pos_path, neg_path = split_grad(path_grad, normalize=[max_pos, max_neg])
    pos_path, neg_path = split_grad(path_grad, normalize=False)
    # pos_no_path, neg_no_path = split_grad(no_path_grad, normalize=max_grad)
    ag_path = (path_grad ** 2).squeeze()

    # Plot subplots
    create_subplots(
        pos_path=pos_path,
        neg_path=neg_path,
        path_im=path_im,
        label=label,
        idx=idx,
        out_path=out_path)

    # Interpolate if requested
    if interp_frames:
        pos_path = interpolate(pos_path, interp_frames)
        neg_path = interpolate(neg_path, interp_frames)
        ag_path = interpolate(ag_path, interp_frames)
        # pos_no_path = interpolate(pos_no_path, interp_frames)
        # neg_no_path = interpolate(neg_no_path, interp_frames)

    if plot_movies:
        # Plot path grad
        fig, ani = subplot_animate_frames(
            image=path_im,
            neg=neg_path,
            pos=pos_path,
            ag=ag_path,
            label=label,
            im_cmap='Greys',
            neg_cmap='Reds',
            pos_cmap='Blues',
            ag_cmap='Greens')
        ani.save(
            os.path.join(
                out_path,
                '%s_%s_subplots.mp4' % (idx, label.replace(' ', '_'))),
            dpi=dpi)

    # # Plot nopath grad
    # fig, ani = subplot_animate_frames(
    #     image=no_path_im,
    #     neg=neg_no_path,
    #     pos=pos_no_path,
    #     im_cmap='Greys',
    #     neg_cmap='Reds',
    #     pos_cmap='Blues')
    # # ani.save('subplots.gif', writer='imagemagick')
    # ani.save('nopath_subplots.mp4', dpi=600)

    # # ani_pos.save('pos.mp4')
    # # plt.close(fig_pos)
    # # fig_neg, ani_neg = animate_frames(neg_path, cmap='Blues')
    # # ani_neg.save('neg.mp4')
    # # plt.close(fig_neg)


# Config
file_path = '/home/drew/Documents/hgru/results'
out_path = 'movies'
f = os.path.join(file_path, 'val_gradients.npz')
im_key = 'val_images'
grad_key = 'val_gradients'
lab_key = 'val_labels'
path_index = 3
nopath_index = 0
interp_frames = 100

# Process data
data = np.load(f)
images = data[im_key]
grads = data[grad_key]
labels = data[lab_key]
py_utils.make_dir(out_path)

# Render movies in a loop
for idx in range(len(images)):
    render_movie(
        images=images,
        grads=grads,
        labels=labels,
        idx=idx,
        interp_frames=interp_frames,
        out_path=out_path)

