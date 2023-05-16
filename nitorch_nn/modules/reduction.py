"""Various reduction layers.

Reductions apply a function across one or several dimensions and collapse them.

(Nan)MaxReduction
(Nan)MinReduction
(Nan)MedianReduction
(Nan)SumReduction
(Nan)MeanReduction
(Nan)VarReduction
(Nan)StdReduction
"""

from nitorch_nn.base import Module
from nitorch_fastmath import reduce


class Reduction(Module):
    """Base class for reductions."""
    reduction = reduce.max
    omitnan = False

    def __init__(self, dim=None, keepdim=False, omitnan=None, reduction=None):
        """

        Parameters
        ----------
        dim : int or list[int], default=`2:`
            Dimensions to reduce. By default, all but batch and channel.
        keepdim : bool, default=False
            Keep reduced dimensions.
        omitnan : bool, default=False
            Discard nans when reducing.
        reduction : str or  callable, optional
            Reduction function
        """

        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.omitnan = omitnan if omitnan is not None else type(self).omitnan
        self.reduction = reduction or type(self).reduction
        if isinstance(self.reduction, str):
            if not self.reduction in reduction_functions:
                raise ValueError(f'Unknown reduction {reduction}')
            self.reduction = reduction_functions[self.reduction]

    def forward(self, x):
        """

        Parameters
        ----------
        x : (batch, channel, *spatial) tensor
            Tensor to reduce

        Returns
        -------
        reduced : tensor
            Reduced tensor

        """
        dim = self.dim
        if dim is None:
            dim = list(range(2, x.dim()))
        if isinstance(dim, slice):
            start = dim.start if dim.start is not None else 2
            stop = dim.stop if dim.stop is not None else x.dim()
            step = dim.step if dim.step is not None else 1
            dim = list(range(start, stop, step))

        return self.reduction(x, dim, keepdim=self.keepdim, omitnan=self.omitnan)


class MaxReduction(Reduction):
    """Maximum"""
    reduction = reduce.max


class MinReduction(Reduction):
    """Minimum"""
    reduction = reduce.min


class MedianReduction(Reduction):
    """Median"""
    reduction = reduce.median


class SumReduction(Reduction):
    """Sum"""
    reduction = reduce.sum


class MeanReduction(Reduction):
    """Mean"""
    reduction = reduce.mean


class VarReduction(Reduction):
    """Mean"""
    reduction = reduce.var


class StdReduction(Reduction):
    """Mean"""
    reduction = reduce.std


class NanMinReduction(MinReduction):
    omitnan = True


class NanMaxReduction(MaxReduction):
    omitnan = True


class NanSumReduction(SumReduction):
    omitnan = True


class NanMeanReduction(MeanReduction):
    omitnan = True


class NanVarReduction(VarReduction):
    omitnan = True


class NanStdReduction(StdReduction):
    omitnan = True


reduction_functions = {
    'max': reduce.max,
    'min': reduce.min,
    'median': reduce.median,
    'sum': reduce.sum,
    'mean': reduce.mean,
    'nanmax': reduce.nanmax,
    'nanmin': reduce.nanmin,
    'nansum': reduce.nansum,
    'nanmean': reduce.nanmean,
}


reductions = {
    'max': MaxReduction,
    'min': MinReduction,
    'median': MedianReduction,
    'sum': SumReduction,
    'mean': MeanReduction,
    'nanmax': NanMaxReduction,
    'nanmin': NanMinReduction,
    'nansum': NanSumReduction,
    'nanmean': NanMeanReduction,
}
