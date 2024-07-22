import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest

class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.GELU(),
            nn.Conv2d(out_channels*2, in_channels, kernel_size=3, padding=1)
        )
        self.out_conv = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, x):
        out = self.network(x)
        out = torch.add(out,x)
        out = self.out_conv(out)
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x_size = x.shape[-1]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.channels, -1).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(batch_size, -1, x_size, x_size)

class Encoder(nn.Module):
    def __init__(self, latent_dim = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResNet(3,128),
            nn.ReLU(),
            ResNet(128,128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResNet(128,256),
            nn.ReLU(),
            ResNet(256,512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResNet(512,256),
            nn.ReLU(),
            ResNet(256,256),
            nn.ReLU(),
            
            nn.Conv2d(256,latent_dim,3)
        )
    
    def forward(self,img):
        out = self.net(img)
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim = 32) -> None:
        super().__init__()
        self.res = ResNet(latent_dim,128)
        self.res1 = ResNet(128,128)
        self.res2 = ResNet(128,256)
        self.res3 = ResNet(256,512)
        self.res4 = ResNet(512,256)
        self.res5 = ResNet(256,256)
        self.up = nn.ConvTranspose2d(128, 128, 3)
        self.up1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1,output_padding=1)
        self.up2 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1,output_padding=1)
        self.out = nn.Conv2d(256,3,1)
        
    
    def forward(self,img):
        out = F.relu(self.res(img))
        out = self.up(out)
        out = F.relu(self.res1(out))
        out = self.up1(out)
        out = F.relu(self.res2(out))
        out = F.relu(self.res3(out))
        out = self.up2(out)
        out = F.relu(self.res4(out))
        out = F.relu(self.res5(out))
        out = self.out(out)
        return out
    
class TestDecoder(unittest.TestCase):
    def test_decoder_output_shape(self):
        # Create a dummy input image with latent representation size
        img = torch.randn(2, 32, 14, 14)
        
        # Initialize the decoder
        decoder = Decoder(latent_dim=32)
        
        # Pass the image through the decoder
        out = decoder(img)
        
        # Check the output shape
        self.assertEqual(out.shape, (2, 3, 64, 64), f"Output shape mismatch: expected (2, 32, 32, 32), got {out.shape}")


if __name__ == '__main__':
    unittest.main()