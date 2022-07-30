import torch
import torch as nn
import segmentation_models_pytorch as smp
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import os


class NC_Net(nn.Module):
    def __init__(self, encoder, encoder_weights, device):
        super().__init__()
        self.model_name = "unet"
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.classes = 3
        self.activaton = None
        self.device = device

        self.model = None
        if self.model_name == "unet":
            self.model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=self.classes - 1,
                activation=self.activaton,
                decoder_attention_type="scse",
            )
            self.model1 = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=self.classes - 2,
                activation=self.activaton,
                decoder_attention_type="scse",
            )
        else:
            print("Model Not Found !")

    def forward(self, x):
        features = self.model.encoder(x)

        decoder_output = self.model.decoder(*features)
        decoder_output1 = self.model1.decoder(*features)

        decoder_output = self.model.segmentation_head(decoder_output)
        decoder_output1 = self.model1.segmentation_head(decoder_output1)

        masks = torch.zeros(
            decoder_output1.size(0),
            self.classes,
            decoder_output1.size(2),
            decoder_output1.size(3),
        ).to(self.device)
        masks[:, :2, :, :] = decoder_output
        masks[:, 2, :, :] = decoder_output1.squeeze()

        return masks

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x
