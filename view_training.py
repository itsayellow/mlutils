#!/usr/bin/env python3
#
# Post-training view of loss, accuracy metrics


import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np


def process_command_line(argv):
    """Process command line invocation arguments and switches.

    Args:
        argv: list of arguments, or `None` from ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: named attributes of arguments and switches
    """
    #script_name = argv[0]
    argv = argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(
            description="Plot training metrics.")

    # specifying nargs= puts outputs of parser in list (even if nargs=1)

    # required arguments
    parser.add_argument('datadir',
            help="Directory containing train_history.json."
            )

    # switches/options:
    #parser.add_argument(
    #    '-s', '--max_size', action='store',
    #    help='String specifying maximum size of images.  ' \
    #            'Larger images will be resized. (e.g. "1024x768")')
    #parser.add_argument(
    #    '-o', '--omit_hidden', action='store_true',
    #    help='Do not copy picasa hidden images to destination directory.')

    args = parser.parse_args(argv)

    return args


def plot_vs_epoch(ax, epochs, train=None, val=None, do_legend=True):
    if train is not None:
        ax.plot(epochs, train, '.-', label='training')
    if val is not None:
        ax.plot(epochs, val, '.-', label='validation')
    if do_legend:
        ax.legend()


def plot_loss(ax, data_dict):
    # ax belongs to fig
    ax.set_title("Loss During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plot_vs_epoch(
            ax,
            range(1, len(data_dict['acc'])+1),
            data_dict.get('loss', []),
            data_dict.get('val_loss', [])
            )
    ax.grid()


def plot_acc(ax, data_dict):
    # ax belongs to fig
    ax.set_title("Accuracy During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    plot_vs_epoch(
            ax,
            range(1, len(data_dict['acc'])+1),
            100*np.array(data_dict.get('acc', [])),
            100*np.array(data_dict.get('val_acc', []))
            )
    ax.grid()


def pyplot_quick_dirty(train_data):
    epochs = range(1, len(train_data['loss'])+1)
    plt.figure(num=1, figsize=(10,5))

    plt.subplot(121)
    plt.plot(epochs, train_data['loss'], label='loss')
    plt.plot(epochs, train_data['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(epochs, 100*np.array(train_data['acc']), label='acc')
    plt.plot(epochs, 100*np.array(train_data['val_acc']), label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()


def plot_loss_acc(fig, train_data):
    plt.subplots_adjust(
            left=None,
            bottom=None,
            right=None,
            top=None,
            wspace=None,
            hspace=None,
            )
    ax1 = fig.add_subplot(121)
    plot_loss(ax1, train_data)
    ax2 = fig.add_subplot(122)
    plot_acc(ax2, train_data)

def main(argv=None):
    args = process_command_line(argv)

    data_dir = pathlib.Path(args.datadir)
    train_data_path = data_dir / 'train_history.json'
    with train_data_path.open("r") as train_data_fh:
        train_data = json.load(train_data_fh)

    # actually plot
    fig = plt.figure(num=1, figsize=(10,5))
    plot_loss_acc(fig, train_data)
    fig.savefig(data_dir.name + ".png", bbox_inches="tight")
    plt.show()

    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv)
    except KeyboardInterrupt:
        # Make a very clean exit (no debug info) if user breaks with Ctrl-C
        print("Stopped by Keyboard Interrupt", file=sys.stderr)
        # exit error code for Ctrl-C
        status = 130

    sys.exit(status)
