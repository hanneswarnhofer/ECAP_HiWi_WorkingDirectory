from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from torch_geometric.nn import DynamicEdgeConv
from torch import Tensor

try:
    from torch_cluster import knn
except ImportError:
    knn = None


class DynamicEdgeConvCoords(DynamicEdgeConv):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(DynamicEdgeConvCoords,
              self).__init__(nn=nn, k=k, num_workers=num_workers, aggr=aggr, **kwargs)

    def forward(
            self, x: Union[Tensor, PairTensor],
            xf: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `DynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=xf, size=None)
