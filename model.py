import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import unittest
from modules import Encoder,Decoder

class VQVAE(nn.Module):
    def __init__(self,latent_dim = 32, num_embeddings = 128, beta = 0.25) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.codebook = nn.Embedding(num_embeddings,latent_dim)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.beta = beta
    
    def forward(self, img):
        # Encode the image
        z = self.encoder(img)
        B, C, H, W = z.shape

        # Reshape the latent representation to (batch_size, height*width, latent_dim)
        z_copy = z.view(z.shape[0], z.shape[1], -1).permute(0, 2, 1)  # Shape: (batch_size, height*width, latent_dim)
        
        # Compute the distances between z and the codebook entries
        codebook_entries = self.codebook.weight  # Shape: (num_embeddings, latent_dim)
        
        # Compute the distances (squared L2 norm)
        distances = torch.cdist(z_copy, codebook_entries.unsqueeze(0), p=2)  # Shape: (batch_size, height*width, num_embeddings)
        
        # Find the nearest codebook entry for each vector in z
        min_distances, indices = torch.min(distances, dim=-1)
       
        # Get the corresponding codebook vectors
        z_q = self.codebook(indices)  # Shape: (batch_size, height*width, latent_dim)

        # Reshape z_q back to the spatial dimensions of z
        z_q = z_q.permute(0, 2, 1).view((B, C, H, W))  # Shape: (batch_size, latent_dim, height, width)
        
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)

        loss = commitment_loss + codebook_loss*self.beta

        out = self.decoder(z)
        return out, loss
    
    def inference(self, batch_size, height=64, width=64,show=True,save=False):
        latent_dim = self.latent_dim
        num_embeddings = self.num_embeddings

        # Generate random indices
        random_indices = torch.randint(0, num_embeddings, (batch_size, height * width))
        
        # Get corresponding embeddings from the codebook
        z_q = model.codebook(random_indices)
        
        # Reshape embeddings to the appropriate shape for the decoder
        z_q = z_q.permute(0, 2, 1).view(batch_size, latent_dim, height, width)
        
        # Generate the output image using the decoder
        output = model.decoder(z_q)

        return output

# Example usage:
# Define the VQVAE model
model = VQVAE(latent_dim=32, num_embeddings=128)
# Create a dummy input image
img = torch.randn(2, 3, 32, 32)
# Pass the image through the model
out,loss = model(img)

print(f'out shape: {out.shape}')
print(f'loss shape: {loss}')

# Generate random output image
output_img = model.inference(1, 6, 6)
print(output_img.shape)  # Expected output: torch.Size([2, 3, output_height, output_width])