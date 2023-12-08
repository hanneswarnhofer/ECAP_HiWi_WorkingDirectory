from models.torch.base import BaseModel, Outputs
from torch_geometric.nn import global_max_pool, EdgeConv, DynamicEdgeConv
from torch import concat
import torch.nn as nn
from models.torch.architectures import get_resnet_tower, get_simple_hfunc

F = nn.functional


class DynEdgeConv(BaseModel):
    def __init__(self, tasks, nb_feat=64, drop=0.5):
        super().__init__(tasks)

        h1_ct14 = get_simple_hfunc(6, nb_feat)
        self.edge_conv1 = EdgeConv(h1_ct14, aggr="mean")
        h2_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.edge_conv2 = EdgeConv(h2_ct14, aggr="mean")
        h3_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.edge_conv3 = EdgeConv(h3_ct14, aggr="mean")
        h4_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.dyn_edge_conv1 = DynamicEdgeConv(h4_ct14, k=10, aggr="mean")
        h5_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.dyn_edge_conv2 = DynamicEdgeConv(h5_ct14, k=10, aggr="mean")
        h6_ct14 = get_simple_hfunc(nb_feat, nb_feat)
        self.dyn_edge_conv3 = DynamicEdgeConv(h6_ct14, k=10, aggr="mean")

        nb_feat2 = 6 * nb_feat
        self.batchnorm = nn.BatchNorm1d(nb_feat2)
        # self.drop1 = nn.Dropout(drop)
        self.resnet = get_resnet_tower(nb_feat2, nb_feat, 3, activation=nn.SiLU, bn=True, dropout_fraction=drop)

        self.out = Outputs(n_inputs=nb_feat, tasks=tasks)

    def get_batch(self, inputs):
        return [inputs["ct%i_batch" % i] for i in range(1, 5)]

    def forward(self, inputs={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ei_pos_ct2": None,
                              "ei_pos_ct3": None, "ei_pos_ct4": None, "pos_ct1": None,
                              "pos_ct2": None, "pos_ct3": None, "pos_ct4": None}):

        # alias
        ei = inputs["edge_index"]
        batch = inputs["batch"]

        # Graph Network
        coords_and_signals = concat([inputs["swgo_pos"], inputs["feat"]], axis=-1)
        graph_out = []
        graph_out.append(F.relu(self.edge_conv1(coords_and_signals, ei)))
        graph_out.append(F.relu(self.edge_conv2(graph_out[-1], ei)))
        graph_out.append(F.relu(self.edge_conv3(graph_out[-1], ei)))
        graph_out.append(F.relu(self.dyn_edge_conv1(graph_out[-1], batch)))
        graph_out.append(F.relu(self.dyn_edge_conv2(graph_out[-1], batch)))
        graph_out.append(F.relu(self.dyn_edge_conv3(graph_out[-1], batch)))

        pooled = global_max_pool(concat(graph_out, axis=-1), batch)

        x = self.batchnorm(pooled)
        # x = self.drop1(x)
        x = self.resnet(x)
        return self.out(x)
