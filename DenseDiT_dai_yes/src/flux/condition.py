import torch
from typing import Union, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image

from .pipeline_tools import encode_images
import sys


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor] = None,
    ) -> None:
        if condition is None:
            print("Error: Query image is required")
            sys.exit("Query image is required")
        else:
            self.condition = condition

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens and ids.
        """
        tokens, ids = encode_images(pipe, self.condition)
        return tokens, ids
