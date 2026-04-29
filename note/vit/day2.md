Simple ViT
An update from some of the same authors of the original paper proposes simplifications to that allows it to train faster and better.ViT

Among these simplifications include 2d sinusoidal positional embedding, global average pooling (no CLS token), no dropout, batch sizes of 1024 rather than 4096, and use of RandAugment and MixUp augmentations. They also show that a simple linear at the end is not significantly worse than the original MLP head

You can use it by importing the as shown belowSimpleViT

import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
