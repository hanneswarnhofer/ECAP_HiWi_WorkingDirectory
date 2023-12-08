from torch_geometric.data import Data


class MultiGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'edge_index' in key or 'face' in key:
            return 1
        else:
            return 0

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            k = key.split("_edge_index")[0]
            k = "self." + k + "_pos.size(0)"
            return eval(k)
        else:
            return super().__inc__(key, value, *args, **kwargs)
