'''
This script is designed to generate the tokens necessary for training the transformer model.
'''

import torch
import numpy as np
from tqdm import tqdm
import os
from model import VQVAE 
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid

def data_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, max_grad_norm=1.0, epoch_start=0, save_img=True, show_img=False):
    indices_list = []

    for epoch in range(epoch_start, n_epochs):
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)
            indices = model.return_indices(imgs)
            indices_list.append(indices.cpu().numpy())

    return np.concatenate(indices_list)

if __name__ == "__main__":
    path = ''  #path to img dataset
    model_path = ''  #path to vqvae model
    
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VQVAE(latent_dim=2,beta=0.25)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss().to(device)
    print(f"Model parameters: {count_parameters(model)}")

    data_loader = get_data_loader(path, batch_size=64)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    indices = data_loop(
        n_epochs=1,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        epoch_start=0,
    )

    np.save('indices.npy', indices)
    print("Indices saved to indices.npy")

    # Save indices to a text file
    np.savetxt('indices.txt', indices, fmt='%d')
    print("Indices saved to indices.txt")
