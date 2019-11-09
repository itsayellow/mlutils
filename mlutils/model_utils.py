#!/usr/bin/env python3


# python standard packages
import datetime
import functools
import hashlib
import json
import logging
import pkg_resources
import re
import warnings

# import pprint
import sys


# pypi packages
try:
    import tensorflow as tf
except:
    tf = None
import tensorflow.keras.callbacks
from tensorflow.keras.utils import serialize_keras_object
import numpy as np


# logging stuff
#   not necessary to make a handler since we will be child logger
#   we use NullHandler so if no config at top level we won't default to printing
#       to stderr
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def log_or_print(msg, use_logging):
    if use_logging:
        LOGGER.info(msg)
    else:
        print(msg)


# TODO 20090515: this currently does not work properly.  Need to get imports
#   working correctly so if we're not using it we do not import matplotlib
#   but so that it actually does work if we are using it
class MattPlotCallback(tensorflow.keras.callbacks.Callback):
    """Plot as training is ongoing

    DO NOT USE if you do not have matplotlib package installed.
    """

    def __init__(self, do_plot_loss=True, do_plot_acc=True):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("matplotlib is not installed.")
        self.do_plot_loss = do_plot_loss
        self.do_plot_acc = do_plot_acc
        self.legend_printed_acc = False
        self.legend_printed_loss = False

    def on_train_begin(self, logs=None):
        if self.do_plot_loss or self.do_plot_acc:
            plt.ion()
        # Accuracy fig
        if self.do_plot_acc:
            self.fig_acc, self.ax_acc = plt.subplots()
            self.ax_acc.set_title("Accuracy During Training")
            self.ax_acc.set_xlabel("Epoch")
            self.ax_acc.set_ylabel("Accuracy (%)")
            self.ax_acc.grid()
        # Loss fig
        if self.do_plot_loss:
            self.fig_loss, self.ax_loss = plt.subplots()
            self.ax_loss.set_title("Loss During Training")
            self.ax_loss.set_xlabel("Epoch")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.grid()
        # data setup
        self.epochs = []
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epochs.append(epoch)
        if self.do_plot_acc:
            if "acc" in logs:
                self.acc.append(logs["acc"] * 100)
            if "val_acc" in logs:
                self.val_acc.append(logs["val_acc"] * 100)
            plot_vs_epoch(
                self.ax_acc,
                self.epochs[-2:],
                train=self.acc[-2:],
                val=self.val_acc[-2:],
                do_legend=not self.legend_printed_acc,
            )
            if not self.legend_printed_acc:
                self.legend_printed_acc = True
        if self.do_plot_loss:
            if "loss" in logs:
                self.loss.append(logs["loss"])
            if "val_loss" in logs:
                self.val_loss.append(logs["val_loss"])
            plot_vs_epoch(
                self.ax_loss,
                self.epochs[-2:],
                train=self.loss[-2:],
                val=self.val_loss[-2:],
                do_legend=not self.legend_printed_loss,
            )
            if not self.legend_printed_loss:
                self.legend_printed_loss = True
        if self.do_plot_loss or self.do_plot_acc:
            plt.pause(0.001)


class ModelCheckpointLogging(tensorflow.keras.callbacks.ModelCheckpoint):
    """Mod of standard ModelCheckpoint to print to log instead of stdout

    See ../KerasMITLicense for license details concerning some code in this class
    
    Args:
        All standard keras.callbacks.ModelCheckpoint arguments
        logger (logging.Logger): keyword arg only.  logger to print messages
            to instead of stdout
    """

    def __init__(self, *args, **kwargs):
        self.logger = kwargs.pop("logger", None)
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            msg = (
                                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                            if self.logger is not None:
                                self.logger.info(msg)
                            else:
                                print("\n" + msg)
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            msg = "Epoch %05d: %s did not improve from %0.5f" % (
                                epoch + 1,
                                self.monitor,
                                self.best,
                            )
                            if self.logger is not None:
                                self.logger.info(msg)
                            else:
                                print("\n" + msg)
            else:
                if self.verbose > 0:
                    msg = "Epoch %05d: saving model to %s" % (epoch + 1, filepath)
                    if self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print("\n" + msg)
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def plot_acc(hist):
    fig, ax = plt.subplots()
    # ax belongs to fig
    ax.set_title("Accuracy During Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    plot_vs_epoch(
        ax,
        range(1, len(hist.history["loss"]) + 1),
        hist.history.get("acc", []),
        hist.history.get("val_acc", []),
    )


def get_model_full_config(model, remove_names=False):
    """
    """
    # serialize model architecture to string, ensuring consistent key order
    #   by sorting keys
    model_config = model.get_config()

    if remove_names:
        # DEBUG only:
        # pprint.pprint(model_config)

        # remove name from config and each layer, because name is arbitrary
        #   (for hashing)
        # this can be a bit brute-force, don't do this if you really need
        #   the config
        model_config.pop("name", None)
        for layer in model_config["layers"]:
            layer["config"].pop("name", None)
            # only in non-Sequential Models
            layer.pop("name", None)
            # only in non-Sequential Models
            layer.pop("inbound_nodes", None)
        # only in non-Sequential Models
        for layer in model_config.get("input_layers", []):
            layer.pop(0)
        # only in non-Sequential Models
        for layer in model_config.get("output_layers", []):
            layer.pop(0)

        # DEBUG only:
        # pprint.pprint(model_config)

    model_info = {}
    model_info["config"] = model_config
    model_info["loss"] = getattr(model, "loss", "")
    model_info["metrics"] = [
        serialize_keras_object(x) for x in getattr(model, "metrics", {})
    ]
    model_info["optimizer"] = serialize_keras_object(getattr(model, "optimizer", {}))
    # model_opt_name = my_model.optimizer.__class__.__module__ + \
    #        "." + my_model.optimizer.__class__.__name__

    return model_info


def get_model_data_subdir(out_dir, model_name):
    model_name_nohash = re.sub(r"_[0-9a-f]+$", "", model_name)
    datetime_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_out_dir = out_dir / (model_name_nohash + "_" + datetime_str)
    return model_out_dir


def hash_model(model, hash_len=6):
    """Create a hash string based on model config, including compile options

    String is as unique as possible for length of hash_len.

    Uncompiled vs. compiled model hash will change value.

    Arguments:
        model (keras.model.Model): keras model
        hash_len (int): length of returned hex hash string

    Returns:
        (str): hex ([0-9a-f]+) string of length hash_len, corresponding to hash
            of model config
    """
    # serialize model architecture to string, ensuring consistent key order
    #   by sorting keys
    # remove name from each layer, because is arbitrary (don't use for hash)
    # sort_keys is important for hashing consistency!
    model_full_config_json = json.dumps(
        get_model_full_config(model, remove_names=True), sort_keys=True
    )
    model_hash = hashlib.md5(model_full_config_json.encode("utf8"))
    return model_hash.hexdigest()[:hash_len]


def save_summary_to_file(model, model_summary_file, history=None):
    """Write summary of model, optimization and metrics to file

    Args:
        model (keras.models.Model):
        model_summary_file (str or pathlib.Path):
        history (keras.History.history or None):
    """
    with open(str(model_summary_file), "w") as summary_fh:
        model.summary(print_fn=lambda x: print(x, file=summary_fh))
        try:
            model_opt = serialize_keras_object(model.optimizer)["class_name"]
        except AttributeError:
            model_opt = ""
        if model_opt != "":
            print("Optimization: " + model_opt, file=summary_fh)
        print("Loss: " + getattr(model, "loss", ""), file=summary_fh)
        print("Metrics: " + str(getattr(model, "metrics", [])), file=summary_fh)

        if history is not None:
            print("-" * 78, file=summary_fh)
            print(
                "Trained for {0} epochs.".format(len(history["loss"])), file=summary_fh
            )
            if "val_loss" in history:
                print(
                    "Minimum validation loss={0:.4} at epoch {1}".format(
                        np.min(history["val_loss"]), np.argmin(history["val_loss"])
                    ),
                    file=summary_fh,
                )


def plot_vs_epoch(ax, epochs, train=None, val=None, do_legend=True):
    if train is not None:
        ax.plot(epochs, train, "ro-", label="training")
    if val is not None:
        ax.plot(epochs, val, "bo-", label="validation")
    if do_legend:
        ax.legend()


def output_system_summary(use_logging=False):
    log_or_print("Python version: " + sys.version, use_logging)
    if tf is not None:
        log_or_print("Tensorflow version " + tf.__version__, use_logging)

    msg = "Installed packages:\n"
    installed_packages = pkg_resources.working_set
    inst_pkgs = {x.project_name: x.version for x in installed_packages}
    for inst_pkg in sorted(inst_pkgs):
        msg += "    " + inst_pkg + "==" + inst_pkgs[inst_pkg] + "\n"
    log_or_print(msg.rstrip(), use_logging)


def return_also_modelname(func):
    """Function decorator to automatically return model name based on
    function name and hash of model

    Wrapped function returns (existing_return, model_name)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = func(*args, **kwargs)
        model_name = func.__name__ + "_" + hash_model(model, 6)
        return (model, model_name)

    return wrapper
