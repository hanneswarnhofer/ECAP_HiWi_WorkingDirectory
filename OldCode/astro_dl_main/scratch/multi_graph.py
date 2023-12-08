from models.torch.base import BaseModel
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data.data import BaseData
from typing import List, Optional, Union
from torch_geometric.data import Dataset
# from sklearn import neighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

F = nn.functional

ct14 = np.random.randn(20).reshape(10, 2)
ct5 = np.random.randn(30).reshape(15, 2)
# A_ct14 = neighbors.kneighbors_graph(ct14, 5)
# A_ct5 = neighbors.kneighbors_graph(ct5, 5)
#
# ei_ct14, _ = utils.convert.from_scipy_sparse_matrix(A_ct14)
# ei_ct5, _ = utils.convert.from_scipy_sparse_matrix(A_ct5)

y = [ct14.sum() + ct5.sum()]


# # SKlearn clustering
# import sklearn
# A=sklearn.neighbors.kneighbors_graph(x, 2)
# coo_arr = A.tocoo()
# from torch_geometric import utils
#
# edge_index, edge_weight = utils.convert.from_scipy_sparse_matrix(A)

# TORCH clustering
transform = T.KNNGraph(k=3)
g_ct14 = transform(Data(pos=torch.Tensor(ct14), y=torch.Tensor(y)))
g_ct5 = transform(Data(pos=torch.Tensor(ct5), y=torch.Tensor(y)))

g_nx = to_networkx(g_ct14)
fig = plt.figure()
nx.draw_networkx(g_nx, pos=ct14)
fig.savefig("/home/hpc/caph/mppi118h/aiact/ct14.png")

g_nx = to_networkx(g_ct5)
fig = plt.figure()
nx.draw_networkx(g_nx, pos=ct5)
fig.savefig("/home/hpc/caph/mppi118h/aiact/ct54.png")


class HessSparseGraphData(Data):
    def __init__(self, x_ct1, x_ct2, x_ct3, x_ct4, x_ct5, ei_ct1, ei_ct2, ei_ct3, ei_ct4, ei_ct5, primary, energy=None, axis=None, impact=None):
        super().__init__()
        self.primary = primary
        self.energy = energy
        self.axis = axis
        self.impact = impact
        self.ei_ct5 = ei_ct5
        self.ei_ct4 = ei_ct4
        self.ei_ct3 = ei_ct3
        self.ei_ct2 = ei_ct2
        self.ei_ct1 = ei_ct1
        self.x_ct1 = x_ct1
        self.x_ct2 = x_ct2
        self.x_ct3 = x_ct3
        self.x_ct4 = x_ct4
        self.x_ct5 = x_ct5

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'ei_ct14':
            return 1
        elif key == 'ei_ct1':
            return 1
        elif key == 'ei_ct2':
            return 1
        elif key == 'ei_ct3':
            return 1
        elif key == 'ei_ct4':
            return 1
        elif key == 'ei_ct5':
            return 1
        else:
            return 0

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ei_ct1':
            return self.x_ct1.size(0)
        if key == 'ei_ct2':
            return self.x_ct2.size(0)
        if key == 'ei_ct3':
            return self.x_ct3.size(0)
        if key == 'ei_ct4':
            return self.x_ct4.size(0)
        if key == 'ei_ct5':
            return self.x_ct5.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class HessFixedGraphData(HessSparseGraphData):
    def __init__(self, x_ct1, x_ct2, x_ct3, x_ct4, x_ct5, ei_ct14, ei_ct5, primary, energy=None, axis=None, impact=None):
        super().__init__(x_ct1, x_ct2, x_ct3, x_ct4, x_ct5, ei_ct14, ei_ct14, ei_ct14, ei_ct14, ei_ct5, primary, energy, axis, impact)
        # super().__init__(x_ct1=x_ct1, x_ct2=x_ct2, x_ct3=x_ct3, x_ct4=x_ct4, x_ct5=x_ct5, ei_ct1=ei_ct14, ei_ct2=ei_ct14, ei_ct3=ei_ct14, ei_ct4=ei_ct14, ei_ct5=ei_ct5)
        self.ei_ct14 = ei_ct14


edge_index_1 = g_ct14.edge_index
edge_index_2 = g_ct5.edge_index
ct14 = torch.Tensor(ct14)
ct5 = torch.Tensor(ct5)
y = torch.Tensor(y)

inp = {**{"ei_ct%i" % i: edge_index_1 for i in range(1, 5)}, **{"x_ct%i" % i: ct14 for i in range(1, 5)}, "x_ct5": ct5, "ei_ct5": edge_index_2, "primary": y}

inp_fixed = {**{"x_ct%i" % i: ct14 for i in range(1, 5)}, "ei_ct14": edge_index_1, "x_ct5": ct5, "ei_ct5": edge_index_2, "primary": y}

data_sparse = HessSparseGraphData(**inp)
data_fixed = HessFixedGraphData(**inp_fixed)

data_list = [data_fixed for i in range(100)]


class DictCollater(Collater):
    def __call__(self, batch):
        return super().__call__(batch).to_dict()


class DictDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=DictCollater(follow_batch, exclude_keys),
            **kwargs,
        )


loader = DataLoader(data_list, batch_size=7, follow_batch=['x_ct1', 'x_ct2', 'x_ct3', 'x_ct4', 'x_ct5'])
batch = next(iter(loader))


class GCN(BaseModel):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.conv1_ct14 = GCNConv(2, 32)
        self.conv2_ct14 = GCNConv(32, 64)

        self.conv1_ct5 = GCNConv(2, 32)
        self.conv2_ct5 = GCNConv(32, 64)

        self.lin = nn.Linear(320, 1)

    def forward(self, inputs={"x_ct1": None, "x_ct2": None, "x_ct3": None, "x_ct4":
                              None, "x_ct5": None, "ei_ct14": None, "ei_ct5": None}):
        x_ct14 = [F.relu(self.conv1_ct14(data[k], data["ei_ct14"])) for k in ["x_ct1", "x_ct2", "x_ct3", "x_ct4"]]
        x_ct14 = [F.relu(self.conv2_ct14(x, data["ei_ct14"])) for x in x_ct14]
        x_ct5 = F.relu(self.conv1_ct5(data["x_ct5"], data["ei_ct5"]))
        x_ct5 = F.relu(self.conv2_ct5(x_ct5, data["ei_ct5"]))

        x = [global_max_pool(x, batch) for x, batch in zip(x_ct14, [data.x_ct1_batch, data.x_ct2_batch, data.x_ct3_batch, data.x_ct4_batch])]
        x += [global_max_pool(x_ct5, data.x_ct5_batch)]
        conc = torch.concat(x, axis=-1)
        y = self.lin(conc)
        # y = F.relu(y)
        return y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = GCN(tasks=["primary"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
loss = nn.MSELoss()


def to_device(data, device):
    feat, labels = data
    return {k: val.to(device) for k, val in feat.items()}, {k: val.to(device) for k, val in labels.items()}


for epoch in range(200):
    print("epoch:", epoch)
    for i, data in enumerate(loader):
        data.to(device)
        feat, lab = model.batch_to_dict(data)
        # data = to_device(data, device)
        optimizer.zero_grad()
        out = model(data)
        loss_ = loss(out, data.primary[:, None])
        loss_.backward()
        optimizer.step()
        # print(loss_)
