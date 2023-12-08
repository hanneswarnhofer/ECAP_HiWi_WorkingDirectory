""" Example Architecture useful for many experiements"""
import torch.nn as nn

# ####################################
# Architectures, e.g., ResNet, DenseNet
# ####################################


class ResNet(nn.Module):
    def __init__(self, module, dropout_fraction, last_activation):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout_fraction, inplace=True)
        self.last_activation = last_activation()

    def forward(self, inputs):
        x = self.module(inputs) + inputs
        return self.dropout(self.last_activation(x))


def get_resnet_module(n_inputs, n_nodes, activation=nn.ELU, bn=True, dropout_fraction=0.):
    transform = [nn.Linear(n_inputs, n_nodes),
                 activation(),
                 nn.BatchNorm1d(n_nodes),
                 nn.Linear(n_nodes, n_nodes),
                 activation(),
                 nn.BatchNorm1d(n_nodes), ]

    x = tuple(x for x in transform if type(x) != nn.BatchNorm1d or bn is True)
    return ResNet(nn.Sequential(*x), dropout_fraction, last_activation=activation)


def get_resnet_tower(n_inputs, n_nodes, n_layers, activation=nn.ELU, bn=True, dropout_fraction=0.):

    # no Dropout in ResNets! --> https://arxiv.org/abs/1801.05134
    # Only applied after a module!
    x = tuple(get_resnet_module(n_nodes, n_nodes, activation, bn, dropout_fraction) for i in range(n_layers))

    if n_inputs != n_nodes:
        x = (nn.Linear(n_inputs, n_nodes), activation(), nn.BatchNorm1d(n_nodes)) + x
    return nn.Sequential(*x)


def get_resnet_module_cnn1d(n_inputs, n_feat, kernel_size, activation=nn.ELU, bn=True, dropout_fraction=0.):
    transform = [nn.Conv1d(n_inputs, n_feat, kernel_size, padding="same"),
                 activation(),
                 nn.BatchNorm1d(n_feat, kernel_size),
                 nn.Conv1d(n_feat, n_feat, kernel_size, padding="same"),
                 activation(),
                 nn.BatchNorm1d(n_feat, kernel_size), ]

    x = tuple(x for x in transform if type(x) != nn.BatchNorm1d or bn is True)
    return ResNet(nn.Sequential(*x), dropout_fraction, last_activation=activation)


def get_resnet_tower_cnn1d(n_inputs, n_feat, kernel_size, n_layers, activation=nn.ELU, bn=True, dropout_fraction=0.):

    # no Dropout in ResNets! --> https://arxiv.org/abs/1801.05134
    # Only applied after a module!
    x = tuple(get_resnet_module_cnn1d(n_feat, n_feat, kernel_size, activation, bn, dropout_fraction) for i in range(n_layers))

    if n_inputs != n_feat:
        x = (nn.Conv1d(n_inputs, n_feat, kernel_size, padding="same"), activation(), nn.BatchNorm1d(n_feat, kernel_size)) + x
    return nn.Sequential(*x)

# ####################################
# Kernel Function, e.g. for EdgeConv
# ####################################


def get_simple_hfunc(inp_dim, out_dim, activation=nn.SiLU, bn=True):
    transform = [nn.Linear(2 * inp_dim, out_dim, bias=False),
                 nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.9),
                 activation(inplace=True),
                 nn.Linear(out_dim, out_dim, bias=False),
                 nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.9),
                 activation(inplace=True),
                 nn.Linear(out_dim, out_dim, bias=False),
                 nn.BatchNorm1d(out_dim, eps=1e-05, momentum=0.9),
                 activation(inplace=True)]

    x = tuple(x for x in transform if type(x) != nn.BatchNorm1d or bn is True)
    return nn.Sequential(*x)
