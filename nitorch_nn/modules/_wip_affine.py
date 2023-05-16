import math as pymath
import torch
from torch import nn as tnn
from nitorch import core, spatial
from nitorch.core import py, utils, linalg
from .cnn import UNet2
from nitorch.nn.base import Module
from .spatial import GridPull, GridResize, AffineGrid, AffineClassic, AffineClassicInverse


class DenseToAffine(Module):
    """Convert a dense displacement field to an affine matrix"""

    def __init__(self, shift=True):
        """

        Parameters
        ----------
        shift : bool, default=True
            Apply a shift so that the center of rotation is in the
            center of the field of view.
        """
        super().__init__()
        self.shift = shift

    def forward(self, grid, weights=None):
        """

        Parameters
        ----------
        grid : (N, *spatial, dim) tensor
            Displacement grid
        weights : (N, *spatial) tensor, optional
            Weights

        Returns
        -------
        aff : (N, dim+1, dim+1)
            Affine matrix that is closest to grid in the least square sense

        """
        shift = self.shift
        backend = dict(dtype=grid.dtype, device=grid.device)
        nb_dim = grid.shape[-1]
        shape = grid.shape[1:-1]
        dim = grid.shape[-1]

        if weights is not None:
            w = weights.reshape([weights.shape[0], -1, 1])
            sumw = w.sum()
        else:
            w = None
            sumw = grid[1:-1].numel()

        eye = torch.eye(dim+1, **backend)
        eye[-1, -1] = 0

        # the forward model is:
        #   phi(x) = M\A*M*x
        # where phi is a *transformation* field, M is the shift matrix
        # and A is the affine matrix.
        # We can decompose phi(x) = x + d(x), where d is a *displacement*
        # field, yielding:
        #   d(x) = M\A*M*x - x = (M\A*M - I)*x := B*x
        # If we write `d(x)` and `x` as large vox*(dim+1) matrices `D`
        # and `G`, we have:
        #   D = G*B'
        # Therefore, the weighted least squares B is obtained as:
        #   B' = inv(G'*W*G) * (G'*W*D)
        # where W is a diagonal matrix of weights. Then, A is
        #   A = M*(B + I)/M
        #
        # Finally, we project the affine matrix to its tangent space:
        #   prm[k] = <log(A), B[k]>
        # were <X,Y> = trace(X'*Y) is the Frobenius inner product.

        def igg(identity):
            # Compute inv(g*W*g'), where g has homogeneous coordinates.
            #   Instead of appending ones, we compute each element of
            #   the block matrix ourselves:
            #       [[g'*W*g,   g'*W*1],
            #        [1'*W*g,   1'*W*1]]
            #    where 1'*W*1 = N_w, the total weight (= N if no weight).
            g = identity.reshape([identity.shape[0], -1, nb_dim])
            gw = g*w if w is not None else g
            gg = g.new_zeros([len(g), dim+1, dim+1])
            sumg = gw.sum(dim=1)
            gg[:, :dim, :dim] = g.transpose(-1, -2).matmul(gw)
            gg[:, :dim, -1] = sumg
            gg[:, -1, :dim] = sumg
            gg[:, -1, -1] = sumw
            return gg.inverse()

        def gd(identity, disp):
            # compute g'*W*d, where g and d have homogeneous coordinates.
            #       [[g'*W*d,   g'*W*1],
            #        [1'*W*d,   1'*W*1]]
            g = identity.reshape([identity.shape[0], -1, nb_dim])
            d = disp.reshape([disp.shape[0], -1, nb_dim])
            gw = d*w if w is not None else g
            dw = d*w if w is not None else d
            sumg = gw.sum(dim=1)
            sumd = dw.sum(dim=1)
            gd = g.new_zeros([len(g), dim+1, dim+1])
            gd[:, :dim, :dim] = g.transpose(-1, -2).matmul(dw)
            gd[:, :dim, -1] = sumg
            gd[:, -1, :dim] = sumd
            gd[:, -1, -1] = sumw
            return gd

        identity = spatial.identity_grid(shape, **backend)[None, ...]
        affine = igg(identity).matmul(gd(identity, grid))
        affine = affine.transpose(-1, -2) + eye

        if shift:
            affine_shift = torch.eye(dim+1, **backend)
            affine_shift[:dim, -1] = torch.as_tensor(shape, **backend)
            affine_shift[:dim, -1].sub(1).div(2).neg()
            affine = spatial.affine_matmul(affine, affine_shift)
            affine = spatial.affine_lmdiv(affine_shift, affine)

        return affine


class AffineMorphFromDense(Module):

    def __init__(self, dim, unet=None, affine='similitude', pull=None, *, in_channels=2):
        super().__init__()

        # default parameters
        unet = dict(unet or {})
        unet.setdefault('encoder', [16, 32, 32, 32, 32])
        unet.setdefault('decoder', [32, 32, 32, 32, 16, 16])
        unet.setdefault('kernel_size', 3)
        unet.setdefault('pool', None)
        unet.setdefault('unpool', None)
        unet.setdefault('activation', tnn.LeakyReLU(0.2))
        pull = dict(pull or {})
        pull.setdefault('interpolation', 1)
        pull.setdefault('bound', 'dct2')
        pull.setdefault('extrapolate', False)

        affine = affine[0].lower()
        liegroup = ('T'    if affine == 't' else  # translation
                    'SE'   if affine == 'r' else  # rigid
                    'CSO'  if affine == 's' else  # similitude
                    'Aff+' if affine == 'a' else  # affine
                    'SE')

        # prepare layers
        super().__init__()
        self.unet = UNet2(dim, in_channels, dim+1, **unet,)
        self.from_dense = DenseToAffine()
        self.to_prm = AffineClassicInverse(liegroup, logzooms=False)
        self.from_prm = AffineClassic(dim, liegroup, logzooms=False)
        self.to_dense = AffineGrid(shift=True)
        self.pull = GridPull(**pull)
        self.dim = dim

        # register losses/metrics
        self.tags = ['image', 'prm', 'segmentation']

    def forward(self, source, target, source_seg=None, target_seg=None,
                *, _loss=None, _metric=None):

        shape = source.shape[2:]
        source_and_target = torch.cat([source, target], dim=1)
        dense = self.unet(source_and_target)
        weights = dense[:, -1].abs_()
        dense = dense[:, :-1]
        dense = utils.channel2last(dense)
        affine = self.from_dense(dense, weights)
        prm = self.to_prm(affine)
        affine = self.from_prm(prm)
        grid = self.to_dense(affine, shape=shape)
        deformed_source = self.pull(source, grid)

        if source_seg is not None:
            deformed_source_seg = self.pull(source_seg, grid)
        else:
            deformed_source_seg = None

        # compute loss and metrics
        self.compute(_loss, _metric,
                     image=[deformed_source, target],
                     prm=[prm],
                     segmentation=[deformed_source_seg, target_seg])

        if source_seg is None:
            return deformed_source, prm
        else:
            return deformed_source, deformed_source_seg, prm



