"""Fixed input maps shared by quantum classifiers and classical controls."""
from __future__ import annotations

from math import isqrt, prod
import torch
import torch.nn.functional as F


def image_conv_grid(image_shape: tuple[int, int], n_features: int) -> tuple[int, int]:
    """Choose the closest-to-square factor grid, with no cropped input pixels.

    For 28x28 images and eight features this is (2, 4), hence a fixed
    14x7 box kernel with stride (14, 7). No weights or layout are learned.
    """
    if len(image_shape) != 2 or min(image_shape) < 1 or n_features < 1:
        raise ValueError('Positive image dimensions and feature count are required')
    h, w = image_shape
    for gh in range(isqrt(n_features), 0, -1):
        if n_features % gh == 0:
            gw = n_features // gh
            if h % gh == 0 and w % gw == 0:
                return gh, gw
    raise ValueError('Feature grid must exactly tile the image without cropping')


def spatial_box_conv2d(x: torch.Tensor, image_shape: tuple[int, int],
                       n_features: int = 8) -> torch.Tensor:
    """One fixed averaging convolution, nonoverlapping stride, valid padding.

    Input is a batch of flat images or (batch, height, width). Output is a
    batch of row-major spatial-grid features. Dtype/device/gradients are kept.
    """
    if x.ndim == 2 and x.shape[-1] == prod(image_shape):
        images = x.reshape(-1, *image_shape)
    elif x.ndim == 3 and tuple(x.shape[-2:]) == tuple(image_shape):
        images = x
    else:
        raise ValueError('Expected flat or native image batches matching image_shape')
    gh, gw = image_conv_grid(image_shape, n_features)
    kh, kw = image_shape[0] // gh, image_shape[1] // gw
    kernel = x.new_full((1, 1, kh, kw), 1.0 / (kh * kw))
    return F.conv2d(images[:, None], kernel, stride=(kh, kw)).flatten(1)


def angle_input_features(x: torch.Tensor, n_features: int,
                         image_shape: tuple[int, int] | None = None) -> torch.Tensor:
    """Spatial convolution for explicitly identified images; legacy 1D pool otherwise."""
    if image_shape is not None:
        return spatial_box_conv2d(x, image_shape, n_features)
    width = x.shape[-1] // n_features
    if width < 1:
        raise ValueError('Input must have at least n_features entries')
    trimmed = x[..., :width * n_features]
    return trimmed.reshape(*x.shape[:-1], n_features, width).mean(dim=-1)
