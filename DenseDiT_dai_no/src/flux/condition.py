import torch
from typing import Optional, Union, List, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import sys

from .pipeline_tools import encode_images

condition_dict = {
    "densedit": 1
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        context: Union[Image.Image, torch.Tensor] = None,
        mask=None,
        position_delta=None,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if condition is None:
            print("Error: Query image is required")
            sys.exit("Query image is required")
        else:
            self.condition = condition
            #TODO: separate the condition and context
            self.context = context
        self.position_delta = position_delta
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[condition_type]

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in [
            "densedit"
        ]:
            tokens, ids = encode_images(pipe, self.condition)
            context_tokens, context_ids = encode_images(pipe, self.context)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, context_tokens, context_ids, type_id
