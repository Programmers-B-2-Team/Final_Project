from collections import namedtuple
from typing import Any, Dict, Tuple, Union
from torch import nn
from pathlib import Path
from u2net import U2NET
from segmentation_models_pytorch import Unet
import torch
from utils import *


model = namedtuple("model", ["path", "model"])
models = {
    "Unet_cloth": model(
        path=Path('./pretrained_models/unet_cloth_seg.pth'),
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    ),
    "Unet_human": model(
        path=Path('./pretrained_models/u2net_human_seg.pth'),
        model=U2NET(3, 1),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    model_dir = models[model_name].path

    if model_name == "Unet_cloth":
        state_dict = torch.load(model_dir, map_location="cpu")["state_dict"]
        state_dict = rename_layers(state_dict, {"model.": ""})
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    return model