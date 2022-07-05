from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from .context_block import context_block2d
from . import config
from .device import device


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True).to(device)
        self.features = self.vgg.features
        self.out_channels = None

        self.conv_blk1 = self.features[0:4]
        self.conv_blk2 = self.features[4:9]
        self.conv_blk3 = self.features[9:16]
        self.conv_blk4 = self.features[16:23]
        self.conv_blk5 = self.features[23:29]

        self.freeze_vgg_bottom_layers()

    def unfreeze_layers(self, from_layer, to_layer):
        for layer in list(self.features)[from_layer: to_layer]:
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = True

    def freeze_vgg_bottom_layers(self):
        for layer in (list(self.conv_blk1) + list(self.conv_blk2) + list(self.conv_blk3)):
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False

    def vgg_weight_init_upper_layers(self):
        for layer in list(self.feature_extraction.children())[9:]:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def unfreeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv_blk1(x)
        x = self.conv_blk2(x)
        x = self.conv_blk3(x)
        x = self.conv_blk4(x)
        x = self.conv_blk5(x)
        return x


if __name__ == "__main__":
    fe = FeatureExtractor().to(device)
    x = torch.randn((1, 3, config.input_height, config.input_width)).to(device)
    print(fe(x).shape)
    # for i, layer in enumerate(list(fe.feature_extraction.children())):
    #     x = layer(x)
    #     print(f"-------------{i}-------------")
    #     print(x.shape)
    #     print(layer)
