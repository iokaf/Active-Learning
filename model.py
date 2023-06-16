
from typing import Dict
import timm
import torch
from torch import nn

import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config: Dict): 
                 
                 #name="resnet50", lr=LR, loss=None):
        super().__init__()

        self.name = config["model"]["name"]
        pretrained = config["model"]["pretrained"]

        self.lr = config["training"]["learning-rate"]

        labels = config["labels"]["names"]

        self.model = timm.create_model(
            self.name, 
            pretrained=True, 
            num_classes=len(labels),
        )

        self.criterion = nn.CrossEntropyLoss()
        
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.model(x)
        return x
