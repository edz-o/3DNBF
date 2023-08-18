from typing import Union

import torch
# from pytorch3d.common.types import Device
from pytorch3d.renderer.utils import TensorProperties

class AmbientLights(TensorProperties):
    """
    A light object representing the same color of light everywhere.
    By default, this is white, which effectively means lighting is
    not used in rendering.
    """

    def __init__(self, *, ambient_color=None, device: str = "cpu") -> None:
        """
        If ambient_color is provided, it should be a sequence of
        triples of floats.

        Args:
            ambient_color: RGB color
            device: Device (as str or torch.device) on which the tensors should be located

        The ambient_color if provided, should be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        """
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0),)
        super().__init__(ambient_color=ambient_color, device=device)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        return torch.zeros_like(points)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros_like(points)