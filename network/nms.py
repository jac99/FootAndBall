# Non-maxima suppression
# Code taken from Kornia library: https://github.com/kornia/kornia/blob/master/kornia/geometry/subpix/nms.py

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_nms_kernel2d(kx: int, ky: int) -> torch.Tensor:
    """Utility function, which returns neigh2channels conv kernel"""
    numel: int = ky * kx
    center: int = numel // 2
    weight = torch.eye(numel)
    weight[center, center] = 0
    return weight.view(numel, 1, ky, kx)


class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size: Tuple[int, int]):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.padding: Tuple[int, int, int, int] = self._compute_zero_padding2d(kernel_size)
        self.kernel = _get_nms_kernel2d(*kernel_size)

    @staticmethod
    def _compute_zero_padding2d(kernel_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return pad(ky), pad(ky), pad(kx), pad(kx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4, x.shape
        B, CH, H, W = x.size()
        # find local maximum values
        max_non_center = F.conv2d(F.pad(x, list(self.padding)[::-1], mode='replicate'),
                                  self.kernel.repeat(CH, 1, 1, 1).to(x.device, x.dtype),
                                  stride=1, groups=CH).view(B, CH, -1, H, W).max(dim=2)[0]
        mask = x > max_non_center
        return x * (mask.to(x.dtype))
