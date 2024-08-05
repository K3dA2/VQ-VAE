import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.network(x)
        return torch.add(out, self.residual_layer(x))

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
    def __init__(self, latent_dim=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResNet(512, 256),
            nn.ReLU(),
            ResNet(256, 256),
            nn.ReLU(),
            ResNet(256, 64),
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        )
    
    def forward(self, img):
        out = self.net(img)
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(64, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNet(128, 128),
            nn.Upsample(scale_factor=2),
            ResNet(128, 32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, img):
        out = self.net(img)
        return out

class TestDecoder(unittest.TestCase):
    def test_decoder_output_shape(self):
        # Create a dummy input image with latent representation size
        img = torch.randn(2, 32, 16, 16)
        
        # Initialize the decoder
        decoder = Decoder(latent_dim=32)
        
        # Pass the image through the decoder
        out = decoder(img)
        
        print(out.shape)
        # Check the output shape
        self.assertEqual(out.shape, (2, 3, 64, 64), f"Output shape mismatch: expected (2, 3, 64, 64), got {out.shape}")

if __name__ == '__main__':
    unittest.main()
