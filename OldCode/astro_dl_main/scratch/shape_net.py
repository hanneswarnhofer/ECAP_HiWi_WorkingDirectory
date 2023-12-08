from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], transform=T.KNNGraph(k=3))
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], transform=T.KNNGraph(k=3))
g = dataset[0]
print(g)
loader = DataLoader(dataset, batch_size=5)
u = next(iter(loader))

g_nx = to_networkx(g)

fig = plt.figure()
nx.draw_networkx(g_nx)
fig.savefig("/home/hpc/caph/mppi118h/aiact/test_graph.png")
