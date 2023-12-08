import os
import numpy as np
from tools import utils
from tools.progress import Progbar
from models.evaluate import Predictor
from data.data import DataContainer
from inspect import signature


class Trainer():
    def __init__(self, model, tasks, log_dir="./", lr=1E-3, epochs=100, name="my_trainer",
                 loss_weights=None, validation_split=0.1, test_split=0.15, batch_size=64,
                 plateau_patience=5, decay_factor=0.33):
        self.model = model
        self.name = name
        # self.tasks = [o.name.split("/")[0] for o in self.model.outputs]
        self.tasks = tasks
        self.lr = lr
        self.epochs = epochs
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.set_loss_weights(loss_weights)
        self.verbose = 1 if utils.is_interactive() else 2
        self.callbacks = None
        self.predictor = Predictor(self.batch_size)
        self.metric_dict, self.loss_dict = {}, {}
        self.plateau_patience = plateau_patience
        self.decay_factor = decay_factor

        # add assertion for checking model outputs and tasks
        import models
        if isinstance(model, models.torch.base.BaseModel):
            self.dtype = "torch"
        elif isinstance(model, models.tf.base.BaseModel):
            self.dtype = "tf"
        else:
            raise TypeError("Model type not supported. Has to be of type 'torch' or 'tf'.")

    @property
    def task_names(self):
        return [t.name for t in self.tasks]

    def config(self):
        return self.get_config()

    def get_config(self):
        config_dict = self.__dict__

        if self.dtype == "tf":
            stringlist = []
            self.model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            return {**config_dict, **{"model": model_summary}}
        else:
            return config_dict

    def save_config(self, log_dir=None):
        log_dir = self.log_dir if log_dir is None else log_dir

        config_dict = self.get_config()
        config = '========================== CONFIG: %s ==========================\n' % self.name

        for k, val in config_dict.items():
            config += str(k) + ": " + str(val) + ",\n"

        with open(os.path.join(self.log_dir, "config"), "w") as f:
            f.write(config)

        return config

    def save(self, log_dir=None, name=None):
        import pickle
        log_dir = self.log_dir if log_dir is None else log_dir
        name = self.name if name is None else name
        dir = os.path.join(log_dir, "%s.pickle" % name)

        with open(dir, 'wb') as file:
            pickle.dump(self, file)

        print("trainer: %s saved to %s" % (name, dir))
        config = self.save_config(log_dir)
        print(config)

    def save_model(self, log_dir=None):
        log_dir = self.log_dir if log_dir is None else log_dir

        os.makedirs(os.path.join(self.log_dir, "save"))

        if self.dtype == "torch":
            ckpt = {'state_dict': self.model.state_dict(), 'batch_size': self.batch_size}
            import torch
            torch.save(ckpt, os.path.join(self.log_dir, "save", "model.pt"))
        elif self.dtype == "tf":
            self.model.save(os.path.join(log_dir, "save"))

        print("model %s saved to %s" % (self.model.name, dir))

    def predict(self, dataset):
        """Function to infer predictions on a dataset using a trained TF/Keras/Torch/PyG model.

        Parameters
        ----------
        dataset : type
            DataContainer that for the evaluation

        Returns
        -------
        predictions : np.array
            Predictions for input dataset

        """
        return self.predictor(self.model, dataset)

    @property
    def metrics(self):
        if self.metric_dict == {}:
            self.set_metrics()

        return self.metric_dict

    def check_function(self, fn, name):
        print(fn, callable(fn))
        assert callable(fn) is True, "Metric / loss has to be a callable function."
        try:
            n_inputs = len(signature(fn).parameters)
        except ValueError:  # use of signature not supported by torch
            n_inputs = len(signature(fn.__call__).parameters)

        assert n_inputs >= 2, "Metric / loss %s has to have at least 2 inputs (y_true / y_pred) but features %i inputs." % (name, n_inputs)

    def set_metrics(self):
        from models.base_metrics import tf_not_implemented, torch_not_implemented

        for learning_task in self.tasks:
            task_list = []

            for metric in learning_task.metrics:
                metric_fn = metric.train_fn(self.dtype)  # callable for torch / string or keras.Metric for tf/keras

                if metric_fn != tf_not_implemented and metric_fn != torch_not_implemented:
                    self.check_function(metric_fn, metric.name)

                    if metric_fn.name not in [t.name for t in task_list]:
                        task_list.append(metric_fn)
                else:
                    print("Metric function of %s not implemented for %s models" % (metric.name, self.dtype))

            self.metric_dict = {**self.metric_dict, learning_task.name: task_list}

    @property
    def losses(self):
        if self.loss_dict == {}:
            self.set_losses()

        return self.loss_dict

    def set_losses(self):
        losses = {}

        for learning_task in self.tasks:
            loss_fn = learning_task.loss(self.dtype)
            self.check_function(loss_fn, learning_task.name)
            losses[learning_task.name] = loss_fn

        self.loss_dict = losses
        return losses

    def set_loss_weights(self, loss_weights=None):
        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = {t_name: 1. for t_name in self.task_names}

    def summary(self):
        return self.model.summary()

    def get_callbacks(self, val_data):
        from keras.callbacks import ReduceLROnPlateau, EarlyStopping
        from models.tf.callbacks import LRTensorBoard, ModelSave, ValidationCallback
        cbacks = []
        cbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=self.decay_factor, patience=self.plateau_patience,
                                        verbose=1, mode='min', min_delta=0.0, min_lr=1.E-5))
        cbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=2 * self.plateau_patience + 1,
                                    verbose=1, mode='min'))
        if val_data is not None:
            cbacks.append(ValidationCallback(val_data, self.batch_size))

        tb = LRTensorBoard(log_dir=os.path.join(self.log_dir, "tensorboard"), histogram_freq=1, write_graph=False,
                           write_images=False, update_freq='epoch')
        cbacks.append(tb)
        cbacks.append(ModelSave(self.log_dir))
        self.callbacks = cbacks
        return cbacks

    def compile(self, **kwargs):
        from tensorflow import keras
        # initial_learning_rate = self.lr
        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=8500,
        #     decay_rate=0.95)

        self.model.compile(keras.optimizers.Adam(self.lr, amsgrad=True, decay=self.get_decay()),
                           loss=self.losses, metrics=self.metrics)

    def train(self, train_data, val_data=None, ad_val_data=None, shuffle=True, interactive=False):
        self.save_config()

        if self.dtype == "torch":
            self.train_torch(train_data, val_data=val_data, shuffle=shuffle, interactive=interactive)
        elif self.dtype == "tf":
            self.train_tf(train_data, val_data=val_data, ad_val_data=ad_val_data, shuffle=shuffle, interactive=interactive)
        else:
            raise NameError("Only training of tf/keras, torch and PyG models implemented.")

    def train_tf(self, train_data, val_data=None, ad_val_data=None, shuffle=True, interactive=False):
        """ Training for tensorflow models using tf.datasets

        Parameters
        ----------
        train_data : tf.dataset(dict)
            training data holding the inputs to the model
        val_data : tf.dataset(dict)
            Description of parameter `val_data`.
        ad_val_data : type
            Description of parameter `ad_val_data`.
        shuffle : bool
            Shuffle dataset before each epoch?

        Returns
        -------
        list
            training history

        """
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.models import Model

        assert isinstance(train_data, DataContainer), "train_data has to be instance of 'DataContainer'"
        assert isinstance(self.model, Model) is True, "For performing Tensorflow training 'train_tf', model has to be tensorflow.keras.models.Model"

        try:
            plot_model(self.model, os.path.join(self.log_dir, "DNN_model.png"), expand_nested=True, show_shapes=True)
        except ImportError:
            print("graphviz not installed on cluster, skip model plotting")

        exclude_keys = self.model.get_exclude_label_keys(train_data.labels.keys())
        tf_train = train_data.tf_loader(self.batch_size, exclude_keys, shuffle=shuffle)

        if val_data is not None:
            assert isinstance(val_data, DataContainer), "val_data has to be instance of 'DataContainer'"
            tf_val = val_data.tf_loader(self.batch_size, exclude_keys, shuffle=False)

        self.iterations = tf_train.cardinality().numpy() / self.batch_size

        self.compile()

        if interactive is True:
            from IPython import embed
            embed()

        history = self.model.fit(tf_train.prefetch(3), validation_data=tf_val, epochs=self.epochs,
                                 callbacks=self.get_callbacks(ad_val_data), verbose=self.verbose)

        self.plot_loss(history.history)
        return history.history

    def plot_loss(self, history, log_dir=None):

        if "loss" not in history.keys():
            return None

        log_dir = self.log_dir if log_dir is None else log_dir
        from matplotlib import pyplot as plt
        epochs = np.arange(len(history["loss"]))

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.plot(epochs, history["loss"], label="loss", c="grey")
        ax.plot(epochs, history["val_loss"], label="val_loss", c="k")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if len(self.tasks) > 1:

            for task, c in zip(self.tasks, colors):
                ax.plot(epochs, history["val_%s_loss" % task.name], label="val_%s" % task.name, c=c)
                ax.plot(epochs, history["%s_loss" % task.name], label="%s_loss" % task.name, c=c, alpha=0.5)

        ax.set_xlabel("Epochs")
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        fig.savefig(os.path.join(self.log_dir, "loss.png"))

    def train_torch(self, train_data, val_data=None, shuffle=True, interactive=False):
        import torch
        import torch.optim as optim
        from torch.utils.tensorboard import SummaryWriter
        from models.torch.base import BaseModel
        assert isinstance(train_data, DataContainer), "train_data has to be instance of 'DataContainer'"
        assert isinstance(self.model, BaseModel) is True, "For performing PyTorch training 'train_torch', model has to be an instance of astro_dl.models.torch.base.BaseModel"

        exclude_keys = self.model.get_exclude_label_keys(train_data.labels.keys())
        trainloader = train_data.torch_loader(self.batch_size, exclude_keys, shuffle=shuffle, follow_batch=self.model.inputs if val_data.dtype == "pyg" else None, drop_last=True)

        if val_data is not None:

            assert isinstance(val_data, DataContainer), "val_data has to be instance of 'DataContainer'"
            valloader = val_data.torch_loader(self.batch_size, exclude_keys, shuffle=False, follow_batch=self.model.inputs if val_data.dtype == "pyg" else None)

        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        metric_dict = self.metrics
        losses = self.losses

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("PyTorch training on device:", device)
        self.model.to(device)
        metric_dict = {k: [v.to(device) for v in val] for k, val in metric_dict.items()}
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor,
                                                         patience=self.plateau_patience, min_lr=1e-5, verbose=1)

        val_loss_min, patience = np.inf, 0
        history = {}

        print(self.model, "\n\n")
        from torchinfo import summary
        summary(self.model)

        if interactive is True:
            from IPython import embed
            embed()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            print("epoch\n%i / %i" % (epoch, self.epochs))
            len_trainloader = trainloader.__len__()
            prog = Progbar(len_trainloader, verbose=self.verbose)
            self.model.train(True) # Change to traning model (BN / Dropout) 

            for i, data in enumerate(trainloader, 0):
                inputs, labels = self.model.batch2dict2device(data)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, task_loss = 0, 0
                prog_dict = {}

                for task in self.tasks:
                    t_name = task.name
                    assert t_name in outputs.keys(), ("task %s is not output by the model." % t_name)
                    task_loss = self.loss_weights[t_name] * losses[t_name](outputs[t_name], labels[t_name])
                    loss += task_loss
                    prog_dict["loss"] = loss.item()
                    prog_dict["%s_loss" % t_name] = task_loss.item()

                    for met in metric_dict[t_name]:
                        name = "%s_%s" % (t_name, met._get_name())
                        met.reset()
                        met(outputs[t_name], labels[t_name])
                        prog_dict[name] = met.compute().item()

                loss.backward()
                optimizer.step()
                prog.add(1, values=list(prog_dict.items()))

            for k, val in prog._values.items():
                val_store = val[0] / len_trainloader
                writer.add_scalar('%s/train' % k, val_store, epoch)

                if k not in history.keys():
                    history[k] = [val_store]
                else:
                    history[k].append(val_store)

            writer.flush()

            if valloader is not None:
                len_valloader = valloader.__len__()
                vprog = Progbar(len_valloader, verbose=self.verbose)
                self.model.eval()  # set eval flag for BN and Dropout

                for i, vdata in enumerate(valloader):
                    vinputs, vlabels = self.model.batch2dict2device(vdata)

                    voutputs = self.model.inference(vinputs)
                    vloss, vtask_loss = 0, 0
                    prog_dict = {}

                    for task in self.tasks:
                        t_name = task.name
                        vtask_loss = self.loss_weights[t_name] * losses[t_name](voutputs[t_name], vlabels[t_name])
                        vloss += vtask_loss
                        prog_dict["val_loss"] = vloss.item()
                        prog_dict["val_%s_loss" % t_name] = vtask_loss.item()

                        for met in metric_dict[t_name]:
                            met.reset()
                            name = "val_%s_%s" % (t_name, met._get_name())
                            met(voutputs[t_name], vlabels[t_name])
                            prog_dict[name] = met.compute().item()

                    vprog.add(1, values=list(prog_dict.items()))

                for k, val in vprog._values.items():
                    val_store = val[0] / len_valloader
                    writer.add_scalar('%s/val' % k, val_store, epoch)

                    if k not in history.keys():
                        history[k] = [val_store]
                    else:
                        history[k].append(val_store)

                scheduler.step(vprog._values["val_loss"][0])
                writer.flush()
                # create checkpoint variable and add important data
                checkpoint = {'epoch': epoch,
                              'valid_loss_min': vprog._values["val_loss"][0],
                              'state_dict': self.model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'batch_size': self.batch_size,
                              }

                if patience > 2 * self.plateau_patience + 1:
                    print("EarlyStopping. val_loss did not decrease after 11 epochs")
                    break

                if vprog._values["val_loss"][0] < val_loss_min:
                    try:
                        torch.save(checkpoint, os.path.join(self.log_dir, "save", "chkpt.pt"))
                    except FileNotFoundError:
                        os.makedirs(os.path.join(self.log_dir, "save"))
                        torch.save(checkpoint, os.path.join(self.log_dir, "save", "chkpt.pt"))

                    val_loss_min = vprog._values["val_loss"][0]
                    patience = 0
                else:
                    patience += 1

        self.plot_loss(history)
        # load best validated model
        checkpoint = torch.load(os.path.join(self.log_dir, "save", "chkpt.pt"))
        self.model.load_state_dict(checkpoint['state_dict'])
        return history

    def fit(self, train_data, ad_val_data):  # train without tf datasets
        self.nsamples = train_data.n_samples
        self.iterations = train_data.n_samples / self.batch_size
        self.compile()
        print(self.model.summary())
        self.model.fit(train_data.tuple, epochs=self.epochs, callbacks=self.get_callbacks(ad_val_data), batch_size=self.batch_size, verbose=self.verbose, validation_split=0.1)
        self.save()

    def get_decay(self, factor=0.1):
        """ Calculate decay rate if learning rate should decay by factor X until the expected end of the training"""
        n_iters = self.iterations * self.epochs
        return (1 / factor - 1) / n_iters


def load_trainer(path):
    import pickle
    with open('path', 'rb') as file:
        loaded_trainer = pickle.load(file)
        return loaded_trainer
