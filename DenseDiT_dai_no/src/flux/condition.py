import torch
from typing import  Union, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image

from .pipeline_tools import encode_images


class Condition(object):
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor] = None,
        context: Union[Image.Image, torch.Tensor] = None,
    ) -> None:
        assert condition is not None, "Query image is required"
        self.condition = condition
        self.context = context

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens and ids.
        """
        tokens, ids = encode_images(pipe, self.condition)
        context_tokens, context_ids = encode_images(pipe, self.context)
        return tokens, ids, context_tokens, context_ids
