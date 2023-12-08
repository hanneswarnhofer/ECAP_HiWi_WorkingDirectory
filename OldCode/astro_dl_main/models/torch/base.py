import torch
import numpy as np
from torch import nn, Tensor
from torch_geometric.data import Data


def to_device(data, device=None):
    if device is None:
        device = "cpu"

    feat, labels = data

    for k, val in feat.items():
        feat[k] = val.to(device)

    for k, val in labels.items():
        labels[k] = val.to(device)

    return feat, labels


class EuclideanRenorm(nn.Module):
    def __init__(self):
        """
            Renormalize the output to the euclidean norm.
        """
        super(EuclideanRenorm, self).__init__()

    def forward(self, tensor):
        return tensor / torch.linalg.vector_norm(tensor, axis=-1, keepdims=True)


class StandardRenorm(nn.Module):
    def __init__(self, mu, std):
        """
            Renormalize the output back to physical quantities.

        Args:
            mu (float): mean of label distribution
            std (float): standard deviation of label distribution
        """
        super(StandardRenorm, self).__init__()
        self.std = nn.Parameter(torch.Tensor([float(std)]), requires_grad=False)
        self.mu = nn.Parameter(torch.Tensor([float(mu)]), requires_grad=False)

    def forward(self, tensor):
        return (tensor * self.std + self.mu).flatten()


class MinMaxRenorm(nn.Module):
    def __init__(self, min, max):
        """
            Renormalize the output back to physical quantities.

        Args:
            min (float): min of label distribution
            max (float): max of label distribution
        """
        super(MinMaxRenorm, self).__init__()
        self.min = nn.Parameter(torch.Tensor([float(min)]), requires_grad=False)
        self.max = nn.Parameter(torch.Tensor([float(max)]), requires_grad=False)

    def forward(self, tensor):
        return ((self.max - self.min) * tensor + self.min).flatten()


class Softmax(nn.Module):
    def __init__(self):
        """
            Normalize the output using softmax.
        """
        super(Softmax, self).__init__()

    def forward(self, tensor):
        return nn.functional.softmax(tensor, dim=-1)


class Outputs(nn.Module):
    def __init__(self, n_inputs, tasks):
        """
        Build outputs of the model by considering the learning tasks and its targets.
        Addionally, the output is renomalized to physical quantities, if normalization was set in the task initialization.

        Args:
            n_inputs (_type_): _description_
            tasks (_type_): _description_
        """
        super().__init__()

        self.layers = {}

        for task in tasks:
            task_seq = []
            lin = nn.Linear(n_inputs, task.nodes)
            setattr(self, "%s_%s" % (task.name, lin._get_name()), lin)
            task_seq.append(lin)

            if task.normalize is not False:
                norm = self.get_renorm_layer(task)
                setattr(self, "%s_%s" % (task.name, norm._get_name()), norm)
                task_seq.append(norm)

            self.layers[task.name] = task_seq

        # for task in tasks:
        #     x = (nn.Linear(n_inputs, task.nodes),)
        #     self.layers.append(x)
        #     if task.normalize is not None:
        #         x += (self.get_renorm_layer(task),)
        #         self.layers.append(self.get_renorm_layer(task))

        #     self.layers_dict[task.name] = nn.Sequential(*x)

    def get_renorm_layer(self, task):

        if task.normalize == "euclidean":
            return EuclideanRenorm()
        elif task.normalize == "standard":
            mu, std = np.mean(task.label()), np.std(task.label())
            return StandardRenorm(mu=mu, std=std)
        elif task.normalize == "min_max":
            min_, max_ = np.min(task.label()), np.max(task.label())
            return MinMaxRenorm(min=min_, max=max_)
        elif task.normalize == "softmax":
            return Softmax()
        else:
            raise NameError("normalization: %s not implemented" % task.normalize)

    def forward(self, inputs):
        outputs = {}

        for output_name, layers in self.layers.items():
            x = inputs

            for layer in layers:
                x = layer(x)

            outputs[output_name] = x

        return outputs


class BaseModel(nn.Module):
    ''' Base class for designing (more keras like) NNs using PyTorch within this framework. '''

    def __init__(self, tasks):
        super().__init__()
        assert self.forward.__defaults__ is not None, "please specify model inputs (attribute defaults) in forward function"
        self.inputs = list(self.forward.__defaults__[0].keys())
        self.outputs = [t.name for t in tasks]
        self.tasks = self.outputs
        self.device = "cpu"
        # keras aliases
        self.output_names = self.outputs
        self.input_names = self.inputs

    @property
    def name(self):
        return self._get_name()

    def summary(self):
        from torchinfo import summary
        return summary(self)

    def to(self, x):
        self.device = x
        return super().to(x)

    def to_device(self, data, device=None):
        return to_device(data, self.device if device is None else device)

    def batch2dict2device(self, batch, device=True):
        """ Take a batch of samples from a PyTorch Geometric Dataset and convert it to a dict."""

        if list(map(type, batch)) == [dict, dict]:  # for torch datasets
            return self.to_device(batch) if device is True else batch

        if device is True:
            batch.to(self.device)

        feat, lab = {}, {}
        batch_ = batch.to_dict()

        for k in self.outputs:
            lab[k] = batch_.pop(k)

        feat = batch_
        return feat, lab

    def batch2dict(self, batch):
        """ Take a batch of samples from a PyTorch Geometric Dataset and convert it to a dict."""
        return self.batch2dict2device(batch, False)

    def predict(self, data: Data) -> Tensor:
        return self.inference(data)

    @torch.inference_mode()
    def inference(self, data: Data) -> Tensor:
        return self.forward(data)

    def get_exclude_label_keys(self, data_keys):
        if type(data_keys) == dict:
            data_keys = data_keys.keys()
        return [k for k in data_keys if k not in self.outputs]

    def get_exclude_keys(self, data_keys):
        if type(data_keys) == dict:
            data_keys = data_keys.keys()
        return [k for k in data_keys if k not in self.inputs + self.outputs]

    def nb_outputs(self) -> int:
        """Number of outputs from GNN model."""
        return len(self.outputs)
