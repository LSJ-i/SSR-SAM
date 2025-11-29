# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Any, Optional
from .transformer import TwoWayTransformer, Attention 
from einops import repeat, rearrange

class SSR_Decoder(nn.Module):
    def __init__(
        self,
        original_imgsize = (512,512),
        num_classes: int = 1, # num_cls = classes_number - 1
        image_embed_dim: int = 768,
        image_embed_hw: Tuple[int, int]  = (32,32), 
        transformer_dim: int = 256,
        global_dim: int = 768,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks

          activation (nn.Module): the type of activation to use when
            upscaling masks

        """
        super().__init__()
        self.num_classes = num_classes
        self.original_size = original_imgsize
        self.transformer_dim = transformer_dim
        self.global_dim = global_dim
        self.image_embed_hw = image_embed_hw
        self.protytpes_num_per_cls = 8
        self.mask_output_embedding = nn.Parameter(torch.randn(1,transformer_dim),requires_grad=True)
        self.prompt_center_prototype = nn.Parameter(torch.randn(self.protytpes_num_per_cls, transformer_dim), requires_grad=True)
        # self.prompt_center_prototype = nn.Parameter(torch.randn(self.num_classes, self.protytpes_num_per_cls, transformer_dim), requires_grad=True)
        self.image_position_embedding = PositionEmbeddingRandom(transformer_dim//2)
        self.transformer = TwoWayTransformer(depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,)
        self.prompt_neck = nn.Linear(self.global_dim,transformer_dim)
        self.prompt_encoder = Attention(transformer_dim, num_heads=8)
        self.neck = nn.Sequential(
            nn.Conv2d(
                image_embed_dim,
                transformer_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
            nn.Conv2d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
        )
        self.output_upscaling = nn.Sequential(
            
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            # LayerNorm2d(transformer_dim // 8),
            activation()
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_classes)
        #     ]
        # )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prototype_prompt: torch.Tensor,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            prompt_embedding=prototype_prompt,
        )

        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        prompt_embedding: torch.Tensor,
    ):  
        image_embeddings = self.window_unpartition(image_embeddings,self.image_embed_hw)
        image_embeddings = self.neck(image_embeddings)
        b, c, h, w = image_embeddings.shape
        image_pe = torch.stack([self.image_position_embedding(self.image_embed_hw) for img in image_embeddings])
        prompt_center_prototype = repeat(self.prompt_center_prototype,'n d -> b nc n d', b=b, nc=prompt_embedding.shape[1])
        # prompt_center_prototype = repeat(self.prompt_center_prototype,'n nc d -> b nc n d', b=b)
        mask_outputs_embeddings = repeat(self.mask_output_embedding,'n d -> b n d', b=b)
        hyper_in_list: List[torch.Tensor] = []
        prompt_embedding = self.prompt_neck(prompt_embedding)
        for cls in range(self.num_classes):
            prompt_queries = self.prompt_encoder(q=prompt_center_prototype[:,cls,:,:], k=prompt_embedding[:,cls,:,:], v=prompt_embedding[:,cls,:,:])
            prompts = torch.concat([mask_outputs_embeddings,prompt_queries],dim=1) # [b, 1+4, d]
            prompt_tokens_out, img_embed_out = self.transformer(image_embeddings, image_pe, prompts)
            mask_outputs = prompt_tokens_out[:,0,:] #[b,d]
            hyper_in_list.append(self.output_hypernetworks_mlps(mask_outputs))
        # prompt_embedding = prompt_embedding.squeeze(2)
        # prompt_tokens_out, img_embed_out = self.transformer(image_embeddings, image_pe, prompt_embedding)
        # prompt_tokens_out, img_embed_out = self.transformer(image_embeddings, image_pe, mask_outputs_embeddings)
        # mask_outputs = prompt_tokens_out
        # masks = []
        # for i in range(self.num_classes):
            # hyper_in_list.append(self.output_hypernetworks_mlps[i](prompt_tokens_out[:, i, :]))
        img_embed_out = img_embed_out.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(img_embed_out) #[b,c,h,w]
        b, c, h, w = upscaled_embedding.shape
        hyper_in = torch.stack(hyper_in_list, dim=1) #[b,nc,d]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # [b,nc,h,w]
        # masks = F.interpolate(masks,self.original_size, mode="bilinear", align_corners=False)

        return masks
    
    def window_unpartition(self,
        windows: torch.Tensor, hw: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
            hw (Tuple): original height and width (H, W) before padding.
        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        H, W = hw
        B = windows.shape[0]
        # x = windows.transpose(2,1).view(B, -1, H, W)
        x = rearrange(windows, 'b (h w) d -> b d h w', h=H)
        return x
    
    def postprocess_masks(
            self,
            masks: torch.Tensor,
            original_size: Tuple[int, ...],
        ) -> torch.Tensor:
            print(masks.shape)
            return F.interpolate(masks, [self.num_classes,original_size[0],original_size[1]], mode="bilinear", align_corners=False)
        
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device : Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



