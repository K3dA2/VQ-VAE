import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision.models as models
from torchvision.transforms.functional import resize
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
from model import VQVAE 
from PIL import Image
import os
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid



def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, data_range=1.0):
    # We'll use SSIM from the torchmetrics library for simplicity
    from torchmetrics.functional import structural_similarity_index_measure
    return structural_similarity_index_measure(img1, img2, data_range=data_range)

# Function to check latent space utilization
def calculate_latent_space_utilization(vqvae, dataloader, device):
    unique_codes = set()
    vqvae.eval()
    
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            # Forward pass through the encoder
            code_indices = vqvae.return_indices(x)
            # Quantize the latent space using the codebook
            # Add the unique indices to the set
            unique_codes.update(code_indices.cpu().numpy().flatten())
    
    utilization = len(unique_codes) / vqvae.num_embeddings
    return utilization


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")



# Load your trained VQ-VAE model
# vqvae = VQVAE(...)
# vqvae.load_state_dict(torch.load('vqvae.pth'))
vqvae = VQVAE(latent_dim = 12, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64)
vqvae.to(device)

val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'
data_loader = get_data_loader(val_path, batch_size = 64, num_samples=10_000)

model_path = 'weights/waifu-vqvae_epoch.pth'

# Optionally load model weights if needed
checkpoint = torch.load(model_path)
vqvae.load_state_dict(checkpoint['model_state_dict'])


psnr_values = []
ssim_values = []
vqvae.eval()

# Compute PSNR and SSIM
with torch.no_grad():
    for x, _ in tqdm(data_loader):
        x = x.to(device)
        recon_x, _ = vqvae(x)
        psnr_values.append(calculate_psnr(x, recon_x).item())
        ssim_values.append(calculate_ssim(x, recon_x).item())

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

# Compute Latent Space Utilization
latent_space_utilization = calculate_latent_space_utilization(vqvae, data_loader, device)

print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Latent Space Utilization: {latent_space_utilization:.4f}")