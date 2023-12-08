from models.torch.base import BaseModel, Outputs
from torch_geometric.nn import GCNConv, global_max_pool, EdgeConv  # , DynamicEdgeConv, global_add_pool
import torch_geometric.nn as gnn
from torch import concat, stack
import torch.nn as nn
from models.torch.architectures import get_resnet_tower, get_simple_hfunc, get_resnet_module_cnn1d

F = nn.functional


class DummyGCN(BaseModel):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.conv1_ct14 = GCNConv(1, 32)
        self.conv2_ct14 = GCNConv(32, 64)

        self.conv1_ct5 = GCNConv(1, 32)
        self.conv2_ct5 = GCNConv(32, 64)

        self.lin = nn.Linear(320, 2)

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4":
                              None, "ct5": None, "ei_ct14": None, "ei_ct5": None}):

        x_ct14 = [nn.SiLU()(self.conv1_ct14(inputs[k], inputs["ei_pos_ct14"])) for k in ["ct1", "ct2", "ct3", "ct4"]]
        x_ct14 = [nn.SiLU()(self.conv2_ct14(x, inputs["ei_pos_ct14"])) for x in x_ct14]
        x_ct5 = nn.SiLU()(self.conv1_ct5(inputs["ct5"], inputs["ei_pos_ct5"]))
        x_ct5 = nn.SiLU()(self.conv2_ct5(x_ct5, inputs["ei_pos_ct5"]))

        x = [global_max_pool(x, batch) for x, batch in zip(x_ct14, [inputs["ct1_batch"], inputs["ct2_batch"], inputs["ct3_batch"], inputs["ct4_batch"]])]
        x += [global_max_pool(x_ct5, inputs["ct5_batch"])]
        conc = concat(x, axis=-1)
        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(conc), dim=-1)

        return outputs


class DummySparseGCN(BaseModel):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.conv1_ct14 = GCNConv(3, 32)
        self.conv2_ct14 = GCNConv(32, 64)
        self.conv3_ct14 = GCNConv(64, 64)

        self.conv1_ct5 = GCNConv(3, 32)
        self.conv2_ct5 = GCNConv(32, 64)
        self.conv3_ct5 = GCNConv(64, 64)

        self.lin = nn.Linear(320, 2)

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ct5":
                              None, "ei_pos_ct1": None, "ei_pos_ct2": None, "ei_pos_ct3": None,
                              "ei_pos_ct4": None, "ei_pos_ct5": None, "pos_ct1": None, "pos_ct2": None, "pos_ct3": None, "pos_ct4": None, "pos_ct5": None}):

        f_ct14 = [concat([inputs["%s" % k], inputs["pos_%s" % k]], axis=-1)
                  for k in ["ct1", "ct2", "ct3", "ct4"]]
        x_ct14 = [nn.SiLU()(self.conv1_ct14(x, inputs["ei_pos_ct%i" % (i + 1)])) for i, x in enumerate(f_ct14)]
        x_ct14 = [nn.SiLU()(self.conv2_ct14(x, inputs["ei_pos_ct%i" % (i + 1)])) for i, x in enumerate(x_ct14)]
        x_ct14 = [nn.SiLU()(self.conv3_ct14(x, inputs["ei_pos_ct%i" % (i + 1)])) for i, x in enumerate(x_ct14)]

        f_ct5 = concat([inputs["ct5"], inputs["pos_ct5"]], axis=-1)
        x_ct5 = nn.SiLU()(self.conv1_ct5(f_ct5, inputs["ei_pos_ct5"]))
        x_ct5 = nn.SiLU()(self.conv2_ct5(x_ct5, inputs["ei_pos_ct5"]))
        x_ct5 = nn.SiLU()(self.conv3_ct5(x_ct5, inputs["ei_pos_ct5"]))

        x = [global_max_pool(x, batch) for x, batch in zip(x_ct14, [inputs["ct1_batch"], inputs["ct2_batch"], inputs["ct3_batch"], inputs["ct4_batch"]])]
        x += [global_max_pool(x_ct5, inputs["ct5_batch"])]
        conc = concat(x, axis=-1)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(conc), dim=-1)

        return outputs


class ResNet(nn.Module):
    def __init__(self, module, dropout_fraction, last_activation):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout_fraction, inplace=True)
        self.last_activation = last_activation()

    def forward(self, inputs):
        x = self.module(inputs) + inputs
        return self.dropout(self.last_activation(x))


def get_fc_tower(n_inputs, n_nodes, n_layers, activation=nn.ELU, dropout_fraction=0.):
    x = tuple()
    for i in range(n_layers):
        x = x + (nn.Linear(n_nodes, n_nodes), activation(), nn.Dropout(dropout_fraction))
    if n_inputs != n_nodes:
        x = (nn.Linear(n_inputs, n_nodes), activation(), nn.Dropout(dropout_fraction)) + x
    return nn.Sequential(*x)


class SparseEdgeConv(BaseModel):
    def __init__(self, tasks, nb_feat=64, drop=0.5):
        super().__init__(tasks)

        h1_ct14 = get_simple_hfunc(3, nb_feat)
        self.conv1_ct14 = EdgeConv(h1_ct14)
        h2_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv2_ct14 = EdgeConv(h2_ct14)
        h3_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv3_ct14 = EdgeConv(h3_ct14)
        h4_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv4_ct14 = EdgeConv(h4_ct14)
        h5_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv5_ct14 = EdgeConv(h5_ct14)
        h6_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv6_ct14 = EdgeConv(h6_ct14)

        h1_ct5 = get_simple_hfunc(3, nb_feat)
        self.conv1_ct5 = EdgeConv(h1_ct5)
        h2_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv2_ct5 = EdgeConv(h2_ct5)
        h3_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv3_ct5 = EdgeConv(h3_ct5)
        h4_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv4_ct5 = EdgeConv(h4_ct5)
        h5_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv5_ct5 = EdgeConv(h5_ct5)
        h6_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv6_ct5 = EdgeConv(h6_ct5)

        nb_feat2 = nb_feat * 6
        self.batchnorm = nn.BatchNorm1d(5 * nb_feat2)
        self.conv1d_1 = nn.Conv1d(5, 16, 1)
        self.conv1d_2 = nn.Conv1d(16, 32, 1)
        self.conv1d_3 = nn.Conv1d(32, 64, 1)
        self.conv1d_4 = nn.Conv1d(64, 1, 1)

        self.lin = nn.Linear(384, 2)

    def get_ei(self, inputs):
        return [inputs["ei_pos_ct%i" % i] for i in range(1, 6)]

    def get_batch(self, inputs):
        return [inputs["ct%i_batch" % i] for i in range(1, 6)]

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ct5":
                              None, "ei_pos_ct1": None, "ei_pos_ct2": None, "ei_pos_ct3": None,
                              "ei_pos_ct4": None, "ei_pos_ct5": None, "pos_ct1": None, "pos_ct2": None, "pos_ct3": None, "pos_ct4": None, "pos_ct5": None}):

        f_ct14 = [concat([inputs["%s" % k], inputs["pos_%s" % k]], axis=-1)
                  for k in ["ct1", "ct2", "ct3", "ct4"]]
        ct_out = []

        for x, ei, b_ in zip(f_ct14, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(nn.SiLU()(self.conv1_ct14(x, ei)))
            out.append(nn.SiLU()(self.conv2_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv3_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv4_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv5_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv6_ct14(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        ct_5 = []
        f_ct5 = concat([inputs["ct5"], inputs["pos_ct5"]], axis=-1)
        ct_5.append(nn.SiLU()(self.conv1_ct5(f_ct5, inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv2_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv3_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv4_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv5_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv6_ct5(ct_5[-1], inputs["ei_pos_ct5"])))

        ct_5 = concat(ct_5, axis=-1)
        ct_out += [global_max_pool(ct_5, inputs["ct5_batch"])]
        conc = stack(ct_out, axis=1)

        conc = self.batchnorm(conc)
        conc = self.conv1d_1(conc)
        conc = nn.ELU()(conc)
        conc = self.conv1d_2(conc)
        conc = nn.ELU()(conc)
        conc = self.conv1d_3(conc)
        conc = nn.ELU()(conc)
        conc = self.conv1d_4(conc)
        conc = nn.ELU()(conc)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(conc.squeeze()), dim=-1)

        return outputs


class SparseEdgeConvStereo(BaseModel):
    def __init__(self, tasks, nb_feat=64, drop=0.5):
        super().__init__(tasks)

        h1_ct14 = get_simple_hfunc(3, nb_feat)
        self.conv1_ct14 = EdgeConv(h1_ct14, aggr="mean")
        h2_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv2_ct14 = EdgeConv(h2_ct14, aggr="mean")
        h3_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv3_ct14 = EdgeConv(h3_ct14, aggr="mean")
        h4_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv4_ct14 = EdgeConv(h4_ct14, aggr="mean")
        h5_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv5_ct14 = EdgeConv(h5_ct14, aggr="mean")
        h6_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv6_ct14 = EdgeConv(h6_ct14, aggr="mean")

        nb_feat2 = nb_feat * 6 * 4
        self.batchnorm = nn.BatchNorm1d(nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_tower(nb_feat2, nb_feat, 5, activation=nn.SiLU, bn=True, dropout_fraction=drop)

        self.lin = nn.Linear(nb_feat, 2)

    def get_ei(self, inputs):
        try:
            return [inputs["ct%i_edge_index" % i] for i in range(1, 5)]
        except KeyError:
            return [inputs["ei_pos_ct14"] for i in range(1, 5)]

    def get_batch(self, inputs):
        return [inputs["ct%i_image_batch" % i] for i in range(1, 5)]

    def forward(self, inputs={"ct1_image": None, "ct2_image": None, "ct3_image": None, "ct4_image": None, "ei_pos_ct2": None,
                              "ei_pos_ct3": None, "ei_pos_ct4": None, "pos_ct1": None,
                              "pos_ct2": None, "pos_ct3": None, "pos_ct4": None}):

        try:
            f_ct14 = [concat([inputs["%s_image" % k], inputs["%s_pos" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4"]]
        except KeyError:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_ct14"]], axis=-1)
                      for k in ["ct1_image", "ct2_image", "ct3_image", "ct4_image"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct14, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(nn.SiLU()(self.conv1_ct14(x, ei)))
            out.append(nn.SiLU()(self.conv2_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv3_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv4_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv5_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv6_ct14(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(x.squeeze()), dim=-1)

        return outputs


class SparseEdgeConvStereoCNN(BaseModel):
    def __init__(self, tasks, nb_feat=64, drop=0.5):
        super().__init__(tasks)

        h1_ct14 = get_simple_hfunc(3, nb_feat)
        self.conv1_ct14 = EdgeConv(h1_ct14, aggr="mean")
        h2_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv2_ct14 = EdgeConv(h2_ct14, aggr="mean")
        h3_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv3_ct14 = EdgeConv(h3_ct14, aggr="mean")
        h4_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv4_ct14 = EdgeConv(h4_ct14, aggr="mean")
        h5_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv5_ct14 = EdgeConv(h5_ct14, aggr="mean")
        h6_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv6_ct14 = EdgeConv(h6_ct14, aggr="mean")

        nb_feat2 = nb_feat * 6 * 4
        self.batchnorm = nn.BatchNorm1d(nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_module_cnn1d(nb_feat2, nb_feat, 5, activation=nn.SiLU, bn=True, dropout_fraction=drop)

        self.lin = nn.Linear(nb_feat, 2)

    def get_ei(self, inputs):
        try:
            return [inputs["ei_pos_ct%i" % i] for i in range(1, 5)]
        except KeyError:
            return [inputs["ei_pos_ct14"] for i in range(1, 5)]

    def get_batch(self, inputs):
        return [inputs["ct%i_batch" % i] for i in range(1, 5)]

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ei_pos_ct1": None, "ei_pos_ct2": None,
                              "ei_pos_ct3": None, "ei_pos_ct4": None, "pos_ct1": None,
                              "pos_ct2": None, "pos_ct3": None, "pos_ct4": None}):

        try:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_%s" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4"]]
        except KeyError:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_ct14"]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct14, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(nn.SiLU()(self.conv1_ct14(x, ei)))
            out.append(nn.SiLU()(self.conv2_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv3_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv4_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv5_ct14(out[-1], ei)))
            out.append(nn.SiLU()(self.conv6_ct14(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(x.squeeze()), dim=-1)

        return outputs


class SparseEdgeConvMono(BaseModel):
    def __init__(self, tasks, nb_feat=64, drop=0.5):
        super().__init__(tasks)

        h1_ct5 = get_simple_hfunc(3, nb_feat)
        self.conv1_ct5 = EdgeConv(h1_ct5, aggr="mean")
        h2_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv2_ct5 = EdgeConv(h2_ct5, aggr="mean")
        h3_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv3_ct5 = EdgeConv(h3_ct5, aggr="mean")
        h4_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv4_ct5 = EdgeConv(h4_ct5, aggr="mean")
        h5_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv5_ct5 = EdgeConv(h5_ct5, aggr="mean")
        h6_ct5 = get_simple_hfunc(nb_feat, nb_feat)
        self.conv6_ct5 = EdgeConv(h6_ct5, aggr="mean")

        nb_feat2 = nb_feat * 6
        self.batchnorm = nn.BatchNorm1d(nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_tower(nb_feat2, nb_feat, 5, activation=nn.SiLU, bn=True, dropout_fraction=drop)
        self.lin = nn.Linear(nb_feat, 2)

    def get_ei(self, inputs):
        return [inputs["ei_pos_ct5"]]

    def get_batch(self, inputs):
        return [inputs["ct5_batch"]]

    def forward(self, inputs={"ct5": None, "ei_pos_ct5": None, "pos_ct5": None}):

        ct_5 = []
        f_ct5 = concat([inputs["ct5"], inputs["pos_ct5"]], axis=-1)
        ct_5.append(nn.SiLU()(self.conv1_ct5(f_ct5, inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv2_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv3_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv4_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv5_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(nn.SiLU()(self.conv6_ct5(ct_5[-1], inputs["ei_pos_ct5"])))

        ct_5 = concat(ct_5, axis=-1)
        ct_out = global_max_pool(ct_5, inputs["ct5_batch"])
        x = self.batchnorm(ct_out)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.lin(x.squeeze()), dim=-1)

        return outputs


class SparseTAGConvMono(BaseModel):
    def __init__(self, tasks, nb_inputs=3, nb_feat=150, nb_hops=2, nb_resnets=2, drop=0.5):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_hops (int): Number of hops 
            nb_feat (int): Number of features in intermediate layer(s)
            nb_resnets (int): Number of resnet layers
            drop (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(tasks)

        # Member variables
        self.nb_inputs = nb_inputs
        self.nb_feat = nb_feat
        self.nb_hops = nb_hops
        self.nb_resnets = nb_resnets
        self.nb_feat2 = 6 * nb_feat

        # Architecture configuration
        self.conv1_ct5 = gnn.TAGConv(self.nb_inputs, self.nb_feat, self.nb_hops)
        self.conv2_ct5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv3_ct5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv4_ct5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv5_ct5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv6_ct5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.batchnorm = nn.BatchNorm1d(self.nb_feat2)

        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_tower(self.nb_feat2, self.nb_feat, self.nb_resnets, activation=nn.SiLU, bn=True, dropout_fraction=drop)
        self.out_primary = nn.Linear(self.nb_feat, 2)
        self.out_energy = nn.Linear(self.nb_feat, 1)
        self.out_axis = nn.Linear(self.nb_feat, 3)
        self.out_impact = nn.Linear(self.nb_feat, 2)

    def get_ei(self, inputs):
        return [inputs["ei_pos_ct5"]]

    def get_batch(self, inputs):
        return [inputs["ct5_batch"]]

    def forward(self, inputs={"ct5": None, "ei_pos_ct5": None, "pos_ct5": None}):

        ct_5 = []
        f_ct5 = concat([inputs["ct5"], inputs["pos_ct5"]], axis=-1)
        ct_5.append(F.selu(self.conv1_ct5(f_ct5, inputs["ei_pos_ct5"])))
        ct_5.append(F.selu(self.conv2_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(F.selu(self.conv3_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(F.selu(self.conv4_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(F.selu(self.conv5_ct5(ct_5[-1], inputs["ei_pos_ct5"])))
        ct_5.append(F.selu(self.conv6_ct5(ct_5[-1], inputs["ei_pos_ct5"])))

        ct_5 = concat(ct_5, axis=-1)
        ct_out = global_max_pool(ct_5, inputs["ct5_batch"])
        x = self.batchnorm(ct_out)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.out_primary(x.squeeze()), dim=-1)

        if "energy" in self.tasks:
            outputs["energy"] = self.out_energy(x)

        if "axis" in self.tasks:
            outputs["axis"] = self.out_axis(x)

        if "impact" in self.tasks:
            outputs["impact"] = self.out_impact(x)

        return outputs


class SparseTAGConvStereo(BaseModel):
    def __init__(self, tasks, nb_inputs=3, nb_feat=150, nb_hops=2, nb_resnets=2, drop=0.5):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_feat (int): Number of nodes in intermediate layer(s)
            nb_hops (int): Number of hops in TAGConv
            nb_resnets (int): Number of resnet layers
            drop (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(tasks)

        # Member variables
        self.nb_inputs = nb_inputs
        self.nb_feat = nb_feat
        self.nb_hops = nb_hops
        self.nb_resnets = nb_resnets
        self.nb_feat2 = 6 * 4 * nb_feat
        self.nb_feat3 = nb_feat

        # Architecture configuration
        self.conv1_ct14 = gnn.TAGConv(self.nb_inputs, self.nb_feat, self.nb_hops)
        self.conv2_ct14 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv3_ct14 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv4_ct14 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv5_ct14 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv6_ct14 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)

        self.batchnorm = nn.BatchNorm1d(self.nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_tower(self.nb_feat2, self.nb_feat3, self.nb_resnets, activation=nn.SiLU, bn=False, dropout_fraction=drop)
        self.out_primary = nn.Linear(self.nb_feat3, 2)
        self.out_energy = nn.Linear(self.nb_feat3, 1)
        self.out_axis = nn.Linear(self.nb_feat3, 3)
        self.out_impact = nn.Linear(self.nb_feat3, 2)

    def get_ei(self, inputs):
        try:
            return [inputs["ei_pos_ct%i" % i] for i in range(1, 5)]
        except KeyError:
            return [inputs["ei_pos_ct14"] for i in range(1, 5)]

    def get_batch(self, inputs):
        return [inputs["ct%i_batch" % i] for i in range(1, 5)]

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None,
                              "ei_pos_ct1": None, "ei_pos_ct2": None, "ei_pos_ct3": None, "ei_pos_ct4": None,
                              "pos_ct1": None, "pos_ct2": None, "pos_ct3": None, "pos_ct4": None}):

        try:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_%s" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4"]]
        except KeyError:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_ct14"]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct14, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(F.selu(self.conv1_ct14(x, ei)))
            out.append(F.selu(self.conv2_ct14(out[-1], ei)))
            out.append(F.selu(self.conv3_ct14(out[-1], ei)))
            out.append(F.selu(self.conv4_ct14(out[-1], ei)))
            out.append(F.selu(self.conv5_ct14(out[-1], ei)))
            out.append(F.selu(self.conv6_ct14(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.out_primary(x.squeeze()), dim=-1)

        if "energy" in self.tasks:
            outputs["energy"] = self.out_energy(x)

        if "axis" in self.tasks:
            outputs["axis"] = self.out_axis(x)

        if "impact" in self.tasks:
            outputs["impact"] = self.out_impact(x)

        return outputs


class SparseTAGConvHybrid(BaseModel):
    def __init__(self, tasks, nb_inputs=3, nb_feat=150, nb_hops=2, nb_resnets=2, drop=0.5):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_feat (int): Number of nodes in intermediate layer(s)
            nb_hops (int): Number of hops in TAGConv
            nb_resnets (int): Number of resnet layers
            drop (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(tasks)

        # Member variables
        self.nb_inputs = nb_inputs
        self.nb_feat = nb_feat
        self.nb_hops = nb_hops
        self.nb_resnets = nb_resnets
        self.nb_feat2 = 6 * 5 * nb_feat
        self.nb_feat3 = nb_feat

        # Architecture configuration
        self.conv1 = gnn.TAGConv(self.nb_inputs, self.nb_feat, self.nb_hops)
        self.conv2 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv3 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv4 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv6 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)

        self.batchnorm = nn.BatchNorm1d(self.nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.fc = get_resnet_tower(self.nb_feat2, self.nb_feat3, self.nb_resnets,
                                   activation=nn.SiLU, bn=False, dropout_fraction=drop)
        self.out_primary = nn.Linear(self.nb_feat3, 2)
        self.out_energy = nn.Linear(self.nb_feat3, 1)
        self.out_axis = nn.Linear(self.nb_feat3, 3)
        self.out_impact = nn.Linear(self.nb_feat3, 2)

    def get_ei(self, inputs):
        try:
            return [inputs["ei_pos_ct%i" % i] for i in range(1, 6)]
        except KeyError:
            return [inputs["ei_pos_ct14"] for i in range(1, 6)]

    def get_batch(self, inputs):
        return [inputs["ct%i_batch" % i] for i in range(1, 6)]

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ct5": None,
                              "ei_pos_ct1": None, "ei_pos_ct2": None, "ei_pos_ct3": None, "ei_pos_ct4": None, "ei_pos_ct5": None,
                              "pos_ct1": None, "pos_ct2": None, "pos_ct3": None, "pos_ct4": None, "pos_ct5": None}):

        try:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_%s" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]
        except KeyError:
            f_ct14 = [concat([inputs["%s" % k], inputs["pos_ct14"]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct14, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(F.selu(self.conv1(x, ei)))
            out.append(F.selu(self.conv2(out[-1], ei)))
            out.append(F.selu(self.conv3(out[-1], ei)))
            out.append(F.selu(self.conv4(out[-1], ei)))
            out.append(F.selu(self.conv5(out[-1], ei)))
            out.append(F.selu(self.conv6(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.fc(x)

        outputs = {}

        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.out_primary(x.squeeze()), dim=-1)

        if "energy" in self.tasks:
            outputs["energy"] = self.out_energy(x)

        if "axis" in self.tasks:
            outputs["axis"] = self.out_axis(x)

        if "impact" in self.tasks:
            outputs["impact"] = self.out_impact(x)

        return outputs


class SparseTAGConvHybridNoTime(BaseModel):
    def __init__(self, tasks, nb_inputs=4, nb_feat=150, nb_hops=2, nb_resnets=2, drop=0.5):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_feat (int): Number of nodes in intermediate layer(s)
            nb_hops (int): Number of hops in TAGConv
            nb_resnets (int): Number of resnet layers
            drop (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(tasks)

        # Member variables
        self.nb_inputs = nb_inputs
        self.nb_feat = nb_feat
        self.nb_hops = nb_hops
        self.nb_resnets = nb_resnets
        self.nb_feat2 = 6 * 5 * nb_feat
        self.nb_feat3 = nb_feat

        # Architecture configuration
        self.conv1 = gnn.TAGConv(self.nb_inputs, self.nb_feat, self.nb_hops)
        self.conv2 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv3 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv4 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv6 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)

        self.batchnorm = nn.BatchNorm1d(self.nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.resnets = get_resnet_tower(self.nb_feat2, self.nb_feat3, self.nb_resnets,
                                        activation=nn.SiLU, bn=False, dropout_fraction=drop)
        self.out = Outputs(n_inputs=nb_feat, tasks=tasks)

    def get_ei(self, inputs):
        try:
            return [inputs["ct%i_edge_index" % i] for i in range(1, 6)]
        except KeyError:
            return [inputs["ct14_edge_index"] for i in range(1, 6)]

    def get_batch(self, inputs):
        return [inputs["ct%i_image_batch" % i] for i in range(1, 6)]

    def forward(self, inputs={"ct1_image": None, "ct2_image": None, "ct3_image": None, "ct4_image": None, "ct5_image": None,
                              "ct1_time": None, "ct2_time": None, "ct3_time": None, "ct4_time": None, "ct5_time": None,
                              "ct1_edge_index": None, "ct2_edge_index": None, "ct3_edge_index": None, "ct4_edge_index": None, "ct5_edge_index": None,
                              "ct1_pos": None, "ct2_pos": None, "ct3_pos": None, "ct4_pos": None, "ct5_pos": None}):

        try:
            f_ct15 = [concat([inputs["%s_image" % k], inputs["%s_pos" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]
        except KeyError:
            f_ct15 = [concat([inputs["%s_image" % k], inputs["ct14_pos"]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct15, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(F.selu(self.conv1(x, ei)))
            out.append(F.selu(self.conv2(out[-1], ei)))
            out.append(F.selu(self.conv3(out[-1], ei)))
            out.append(F.selu(self.conv4(out[-1], ei)))
            out.append(F.selu(self.conv5(out[-1], ei)))
            out.append(F.selu(self.conv6(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.resnets(x)
        return self.out(x)


class SparseTAGConvHybridMultiTask(BaseModel):
    def __init__(self, tasks, nb_inputs=1, nb_feat=150, nb_hops=2, nb_resnets=2, drop=0.5):
        """ConvNet model.
        Args:
            nb_inputs (int): Number of input features, i.e. dimension of input
                layer.
            nb_feat (int): Number of nodes in intermediate layer(s)
            nb_hops (int): Number of hops in TAGConv
            nb_resnets (int): Number of resnet layers
            drop (float): Fraction of nodes to drop
        """
        # Base class constructor
        super().__init__(tasks)

        # Member variables
        self.nb_inputs = nb_inputs
        self.nb_feat = nb_feat
        self.nb_hops = nb_hops
        self.nb_resnets = nb_resnets
        self.nb_feat2 = 6 * 5 * nb_feat
        self.nb_feat3 = nb_feat

        # Architecture configuration
        self.conv1 = gnn.TAGConv(self.nb_inputs, self.nb_feat, self.nb_hops)
        self.conv2 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv3 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv4 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv5 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)
        self.conv6 = gnn.TAGConv(self.nb_feat, self.nb_feat, self.nb_hops)

        self.batchnorm = nn.BatchNorm1d(self.nb_feat2)
        self.drop1 = nn.Dropout(drop)
        self.resnets = get_resnet_tower(self.nb_feat2, self.nb_feat3, self.nb_resnets,
                                        activation=nn.SiLU, bn=False, dropout_fraction=drop)
        self.out = Outputs(n_inputs=nb_feat, tasks=tasks)

    def get_ei(self, inputs):
        try:
            return [inputs["ct%i_edge_index" % i] for i in range(1, 6)]
        except KeyError:
            return [inputs["ct14_edge_index"] for i in range(1, 6)]

    def get_batch(self, inputs):
        return [inputs["ct%i_image_batch" % i] for i in range(1, 6)]

    def forward(self, inputs={"ct1_image": None, "ct2_image": None, "ct3_image": None, "ct4_image": None, "ct5_image": None,
                              "ct1_time": None, "ct2_time": None, "ct3_time": None, "ct4_time": None, "ct5_time": None,
                              "ct1_edge_index": None, "ct2_edge_index": None, "ct3_edge_index": None, "ct4_edge_index": None, "ct5_edge_index": None,
                              "ct1_pos": None, "ct2_pos": None, "ct3_pos": None, "ct4_pos": None, "ct5_pos": None}):

        try:
            f_ct15 = [concat([inputs["%s_image" % k], inputs["%s_time" % k], inputs["%s_pos" % k]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]
        except KeyError:
            f_ct15 = [concat([inputs["%s_image" % k], inputs["%s_time" % k], inputs["ct14_pos"]], axis=-1)
                      for k in ["ct1", "ct2", "ct3", "ct4", "ct5"]]

        ct_out = []

        for x, ei, b_ in zip(f_ct15, self.get_ei(inputs), self.get_batch(inputs)):
            out = []
            out.append(F.selu(self.conv1(x, ei)))
            out.append(F.selu(self.conv2(out[-1], ei)))
            out.append(F.selu(self.conv3(out[-1], ei)))
            out.append(F.selu(self.conv4(out[-1], ei)))
            out.append(F.selu(self.conv5(out[-1], ei)))
            out.append(F.selu(self.conv6(out[-1], ei)))
            ct_out.append(global_max_pool(concat(out, axis=-1), b_))

        x = concat(ct_out, axis=1)
        x = self.batchnorm(x)
        x = self.drop1(x)
        x = self.resnets(x)
        return self.out(x)
