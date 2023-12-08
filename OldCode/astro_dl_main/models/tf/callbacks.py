from tools import utils
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from data.tf import mappings
from functools import partial
import numpy as np
from tools.utils import to_list


class ModelSave(keras.callbacks.Callback):
    def __init__(self, log_dir, monitor="val_loss", epoch_freq=1, **kwargs):
        self.save_dir = utils.create_dir_path(log_dir, "save")
        self.smallest_loss = np.inf
        self.monitor = monitor
        self.epoch_freq = epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor in logs.keys():
            if epoch % self.epoch_freq == 0:
                if logs[self.monitor] < self.smallest_loss:
                    print("val_loss improved from %f to %f, saving model to %s" % (self.smallest_loss, logs["val_loss"], self.save_dir))
                    self.model.save(self.save_dir + '/model_epoch%i.h5' % epoch)
                    self.smallest_loss = logs["val_loss"]


class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:  # lr sheduler used
            steps = self.params["steps"]
            logs.update({'lr_decayed': K.eval(self.model.optimizer.lr((epoch + 1) * steps))})
            # logs.update({'lr': K.eval(self.model.optimizer.learning_rate.initial_learning_rate)})
        except TypeError:  # no lr sheduler used
            logs.update({'lr_decayed': K.eval(self.model.optimizer._decayed_lr(self.model.optimizer.lr.dtype.base_dtype))})
        super().on_epoch_end(epoch, logs)


class ValidationCallback(keras.callbacks.Callback):
    def __init__(self, datasets, batch_size, metrics=None, **kwargs):
        self.metrics = metrics
        self.datasets = to_list(datasets)
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        from data.data import DataContainer

        for i, dset in enumerate(self.datasets):

            try:
                name = dset.name
            except AttributeError:
                name = "_add_%i" % i

            if isinstance(dset, DataContainer) is True:
                if dset.dataset is None:
                    dset = dset.tf()
                else:
                    dset = dset()

            if hasattr(dset, "_batch_size") is False:
                ex_keys = self.model.get_exclude_label_keys(dset.element_spec[1].keys())
                ex_k_fn = partial(mappings.rm_unused_labels, exclude_keys=ex_keys)
                self.datasets[i] = dset.map(ex_k_fn).batch(self.batch_size)
                self.datasets[i].name = name

    def on_epoch_end(self, epoch, logs=None):
        for dset in self.datasets:
            summary = self.model.evaluate(dset, verbose=1 if utils.is_interactive() else 2)
            for m_name, loss in zip(self.model.metrics_names, summary):
                logs["%s_%s" % (dset.name, m_name)] = loss
