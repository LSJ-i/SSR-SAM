import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Any, Optional
from .transformer import TwoWayTransformer, Attention 
from einops import repeat, rearrange

class SSR_model(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        original_imgsize
    ) -> None:
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.original_imgsize = original_imgsize

    def forward(
        self,
        image: torch.Tensor,
        prototype_prompt: torch.Tensor = None,
        only_encoder: bool = False,
        need_fp: bool = False
    ):
        image = torch.stack([self.preprocess(x) for x in image], dim=0)
        image_embeddings = self.encoder(image)

        # image_embeddings = self.encoder.forward_features(image)
        # image_embeddings = image_embeddings[:,1:,:]
        if need_fp:
            image_embeddings = nn.Dropout2d(0.5)(image_embeddings)
            
        masks = self.decoder(image_embeddings, prototype_prompt)
        masks = F.interpolate(masks,(self.encoder.img_size,self.encoder.img_size), mode="bilinear", align_corners=False)
        # masks = F.interpolate(masks,(512,512), mode="bilinear", align_corners=False)
        masks = masks[..., : self.original_imgsize[0], : self.original_imgsize[0]]


        return masks
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    