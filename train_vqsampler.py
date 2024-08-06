import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from transformer_model import VQSampler, Config
from vqsampler_dataloader import get_data_loader
import torch.nn.functional as F
from utils import count_parameters
from model import VQVAE

def training_loop(n_epochs, optimizer, model, device, data_loader, valid_loader, max_grad_norm=1.0, epoch_start=0):
    model.train()
    best_loss_valid = float('inf')  # Initialize best validation loss to infinity
    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        loss_valid = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for valid_tensors, valid_labels in valid_loader:
                valid_tensors = valid_tensors.to(device)
                valid_labels = valid_labels.to(device)
                _, valid_loss = model(valid_tensors, valid_labels)
                loss_valid += valid_loss.item()
        
        loss_valid /= len(valid_loader)

        # Save the model if the validation loss decreases
        if loss_valid < best_loss_valid:
            best_loss_valid = loss_valid
            model_filename = 'vqvae-sampler_best.pth'
            model_path = os.path.join('weights', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

        with open("vqvae-sampler.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")

        with open("vqvae-val-sampler.txt", "a") as file:
            file.write(f"{loss_valid / len(valid_loader)}\n")

        print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(data_loader), loss_valid))

        model.train()

if __name__ == "__main__":
    train_path = ''  # Replace with your actual train data path
    val_path = ''  # Replace with your actual validation data path
    model_path = ''  # Replace with your model weights path
    epoch = 0

    vq_model_path = ''
    vqvae = VQVAE(latent_dim=1, num_embeddings=512, beta=0.25, use_ema=False)  

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # config = Config(num_layers=12, emb_dim=8, num_heads=4, seq_len=256)
    config = Config(num_layers=4, emb_dim=32, num_heads=4, seq_len=256)

    model = VQSampler(config)
    model.to(device)
    vqvae.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)

    print(f"Number of parameters: {count_parameters(model)}")
    data_loader = get_data_loader(train_path, file_type='txt', batch_size=128)
    val_loader = get_data_loader(val_path, file_type='txt', batch_size=128)

    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    '''
    
    training_loop(
        n_epochs=200,
        optimizer=optimizer,
        model=model,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
    )
