#!/usr/bin/env python3
"""
Generic training/testing functions to be used by all experiments
"""

import datetime
import json
import logging

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np

import tictoc

import mlutils.model_utils


# logging stuff
#   not necessary to make a handler since we will be child logger
#   we use NullHandler so if no config at top level we won't default to printing
#       to stderr
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def training(model, epochs, train_x, train_y, val_x, val_y,
        model_out_dir, train_verbose=1, patience=15, batch_size=20,
        realtime_plot=False, resume=None):
    """
    Args:
        model (keras.model): model
        epochs (int): 200
        train_x (numpy.array): data['train']
        train_y (numpy.array): data['train_targets']
        val_x (numpy.array): data['valid']
        val_y (numpy.array): data['valid_targets'],
        model_out_dir (pathlib.Path): model_out_dir,
        train_verbose (int): keras verbose for training (2 for cloud)
        patience (int): how many epochs to tolerate no improvement before
            stopping early
        realtime_plot (bool): use_pyplot
        resume (None or dict):

    Returns:
        model_save_dir: pathlib.Path
    """
    # resume
    if resume is None:
        initial_epoch = 0
    else:
        model = load_model(resume['model_weights_file'])
        # 'data_matt_model1_7f785f/saved_models/weights.final.hd5')
        # initial epoch is last epoch completed before
        initial_epoch = resume['initial_epoch']
        # epochs must be > initial_epoch

    model.summary()

    # Housekeeping
    model_save_dir = model_out_dir / 'saved_models'
    history_save_file = model_out_dir / 'train_history.json'
    model_summary_file = model_out_dir / 'model_summary.txt'
    # create master data dir and saved_models dir under it
    model_save_dir.mkdir(parents=True, exist_ok=True)
    # save summary to file first (in case we crash)
    mlutils.model_utils.save_summary_to_file(model, model_summary_file)

    # Instantiate callbacks
    checkpointer = ModelCheckpoint(
            filepath=str(model_save_dir / 'weights.best.hdf5'),
            verbose=1, save_best_only=True
            )
    checkpointer2 = ModelCheckpoint(
            filepath=str(model_save_dir / 'weights.epoch{epoch:04d}.hdf5'),
            verbose=0, period=20
            )
    early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience
            )
    tensorboard_log_dir = model_out_dir / 'tensorboard'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(
            log_dir = str(tensorboard_log_dir),
            histogram_freq=20,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False,
            update_freq='epoch'
            )
    callbacks = [checkpointer, checkpointer2, early_stopping, tensorboard_callback]
    if realtime_plot:
        mattplot_callback = mlutils.model_utils.MattPlotCallback()
        callbacks.append(mattplot_callback)

    hist = None
    mytimer = tictoc.Timer()
    mytimer.start()

    # Train the model
    try:
        hist = model.fit(
                train_x, train_y,
                validation_data=(val_x, val_y),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=train_verbose,
                initial_epoch=initial_epoch
                )
    except KeyboardInterrupt:
        print("Stopped prematurely via keyboard interrupt.")
    mytimer.eltime_pr("training time: ")

    # save final model+weights
    model.save(str(model_save_dir / 'weights.final.hd5'))

    # save history to json file
    if hist is not None:
        with history_save_file.open("w") as train_hist_fh:
            json.dump(hist.history, train_hist_fh)

    # turn off interactive and hold plots until user dismisses windows
    if realtime_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise Exception("matplotlib is not installed.")
        plt.ioff()
        plt.show()

    # save summary to file with history this time
    mlutils.model_utils.save_summary_to_file(
            model, model_summary_file, history=hist.history
            )

    return model_save_dir


def testing(model, test_x, test_y, model_out_dir, model_save_dir):
    """Testing of experiment

    Args:
        model (keras.model):
        test_x (numpy.array):
        test_y (numpy.array):
        model_out_dir (pathlib.Path):
        model_save_dir (pathlib.Path):
    """
    # Load the model with the best validation loss
    model.load_weights(model_save_dir / 'weights.best.hdf5')

    # get predictions
    predictions = model.predict(test_x)

    # DEBUG
    #print("predictions.shape="+str(predictions.shape))
    #print("predictions[:20, 0]="+str(predictions[:20, 0]))

    # report test accuracy
    if predictions.shape[-1] > 1:
        # multiple-class output
        class_predictions = np.argmax(predictions, axis=1)
        class_actual = np.argmax(test_y, axis=1)
        test_num_correct = np.sum(class_predictions == class_actual)
    else:
        # binary output
        test_num_correct = np.sum((predictions > 0.5) == test_y)

    test_accuracy = test_num_correct/predictions.shape[0]
    test_accuracy_perc = 100 * test_accuracy
    print('Test accuracy: %.4f%%' % test_accuracy_perc)

    test_out = {'test_acc': test_accuracy, 'test_acc_perc': test_accuracy_perc}
    test_out_path = model_out_dir / 'test.json'
    with test_out_path.open("w") as test_out_fh:
        json.dump(test_out, test_out_fh)


def summarize(model, model_name, model_out_dir):
    info_out = {}
    info_out['model_name'] = model_name
    info_out['model_info'] = mlutils.model_utils.get_model_full_config(model)
    info_out['datetime_utc'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    info_out_path = model_out_dir / 'info.json'
    with info_out_path.open("w") as info_out_fh:
        json.dump(info_out, info_out_fh)
