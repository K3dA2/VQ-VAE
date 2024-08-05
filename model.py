import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, Decoder  # Assuming these are correctly implemented

class VQVAE(nn.Module):
    def __init__(self, latent_dim=1, num_embeddings=512, beta=0.25, use_ema=True, ema_decay=0.99):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, latent_dim))
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.register_buffer('usage_count', torch.zeros(num_embeddings))

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
       
        encodings = torch.zeros(indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, indices, 1)
        
        # Get the corresponding codebook vectors
        z_q = self.codebook(indices)  # Shape: (batch_size, height*width, latent_dim)

        # Reshape z_q back to the spatial dimensions of z
        z_q = z_q.permute(0, 2, 1).view((B, C, H, W))  # Shape: (batch_size, latent_dim, height, width)
        
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        if self.use_ema:
            # EMA update for the codebook
            self.ema_inplace_update(indices, z_copy)
            loss = commitment_loss
        else:
            loss = commitment_loss + codebook_loss * self.beta

        # Straight through estimator
        z_q = z + (z_q - z).detach()

        out = self.decoder(z_q)
        self.usage_count += torch.sum(encodings, dim=0)

        return out, loss
    
    def ema_inplace_update(self, indices, flat_inputs):
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_embeddings).float()
            
            ema_cluster_size = torch.sum(encodings, dim=(0, 1))  # Sum across batch and latent dimensions

            # Permute encodings to have shape [num_embeddings, batch_size * num_latents]
            encodings = encodings.permute(2, 0, 1).reshape(self.num_embeddings, -1)
            
            # Reshape flat_inputs to [batch_size * num_latents, latent_dim]
            flat_inputs = flat_inputs.reshape(-1, flat_inputs.shape[-1])
            
            # Perform matrix multiplication
            ema_w = torch.matmul(encodings, flat_inputs)  # Shape: [num_embeddings, latent_dim]

            self.ema_cluster_size.mul_(self.ema_decay).add_(ema_cluster_size, alpha=1 - self.ema_decay)
            self.ema_w.mul_(self.ema_decay).add_(ema_w, alpha=1 - self.ema_decay)

            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n)

            self.codebook.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

    def reset_underused_embeddings(self, img, threshold=1.0):
        z = self.encoder(img)
        underused_indices = self.usage_count < threshold
        if underused_indices.any():
            underused_indices = underused_indices.nonzero(as_tuple=True)[0]
            for idx in underused_indices:
                random_index = torch.randint(0, z.size(0), (1,))
                self.codebook.weight.data[idx] = z[random_index].view(-1).data
            self.usage_count[underused_indices] = threshold  # Reset usage count to prevent immediate re-resetting

    def inference(self, batch_size, height=64, width=64):
        latent_dim = self.latent_dim
        num_embeddings = self.num_embeddings

        # Generate random indices
        random_indices = torch.randint(0, num_embeddings, (batch_size, height * width))
        
        # Get corresponding embeddings from the codebook
        z_q = self.codebook(random_indices)
        
        # Reshape embeddings to the appropriate shape for the decoder
        z_q = z_q.permute(0, 2, 1).view(batch_size, latent_dim, height, width)
        
        # Generate the output image using the decoder
        output = self.decoder(z_q)

        return output
    
    def return_indices(self, img):
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

        return indices

# Example usage:
# Define the VQVAE model
model = VQVAE(latent_dim=1, num_embeddings=512, use_ema=False)
# Create a dummy input image
img = torch.randn(2, 3, 64, 64)
# Pass the image through the model
out, loss = model(img)

print(f'out shape: {out.shape}')
print(f'loss: {loss}')

# Generate random output image
output_img = model.inference(1, 16, 16)
print(output_img.shape)  # Expected output: torch.Size([1, 3, 16, 16])
