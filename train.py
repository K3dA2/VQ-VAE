import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from model import VQVAE 
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid
import uuid



def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, valid_loader,
                  max_grad_norm=1.0, epoch_start=0, save_img=True, show_img=False,
                ema_alpha=0.99,usage_threshold=1.0):

    model.train()
    ema_loss = None
    scheduler = None
    previous_loss = None

    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        mse_loss_train = 0.0
        vq_loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            imgs = imgs.to(device)

            outputs, vq_loss = model(imgs)
            mse_loss = loss_fn(outputs, imgs)
            loss = mse_loss + vq_loss

            loss_train += loss.item()
            mse_loss_train += mse_loss.item()
            vq_loss_train += vq_loss.item()

            loss.backward()

            
            #utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item(), mse_loss=mse_loss.item(), vq_loss=vq_loss.item())

        avg_loss_train = loss_train / len(data_loader)
        avg_mse_loss_train = mse_loss_train / len(data_loader)
        avg_vq_loss_train = vq_loss_train / len(data_loader)

        '''
        # Update EMA of loss
        if ema_loss is None:
            ema_loss = avg_loss_train
        else:
            ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * avg_loss_train

        # Adjust learning rate if average loss increases
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)
        scheduler.step(ema_loss)
        '''

        with open("waifu-vqvae_epoch-loss.txt", "a") as file:
            file.write(f"{avg_loss_train}\n")

        print('{} Epoch {}, Training loss {:.4f}, MSE loss {:.4f}, VQ loss {:.4f}'.format(
            datetime.datetime.now(), epoch, avg_loss_train, avg_mse_loss_train, avg_vq_loss_train))

        if epoch % 5 == 0:
            if save_img:
                with torch.no_grad():
                    for valid_tensors, _ in valid_loader:
                        break

                    save_img_tensors_as_grid(valid_tensors, 4, "true1")
                    val_img, _ = model(valid_tensors.to(device))
                    save_img_tensors_as_grid(val_img, 4, "recon1")

            model_path = os.path.join('./', 'waifu-vqvae_epoch.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            # Reset underused embeddings
        '''
        # Reset underused embeddings conditionally
        if epoch > 1000 and previous_loss is not None and avg_loss_train > previous_loss * 1.25:
            print("reseting")
            with torch.no_grad():
                for batch_imgs, _ in data_loader:
                    model.reset_underused_embeddings(batch_imgs.to(device), threshold=usage_threshold)
                    break

        previous_loss = avg_loss_train
        '''



if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus/Train'
    val_path = '/Users/ayanfe/Documents/Datasets/Waifus/Val'
    model_path = '/Users/ayanfe/Documents/Code/VQ-VAE/weights/waifu-vqvae_epoch.pth'
    epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model = VQVAE(latent_dim=1, num_embeddings=512, beta=0.25, use_ema=False)  # Assuming Unet is correctly imported and defined
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss().to(device)

    print(count_parameters(model))
    data_loader = get_data_loader(path, batch_size=64, num_samples=30_000)
    val_loader = get_data_loader(val_path, batch_size=64, num_samples=10_000)

    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    '''
    
    #print(model.inference(1, 16, 16).shape)
    with torch.no_grad():
        for valid_tensors, _ in val_loader:
            break

        save_img_tensors_as_grid(valid_tensors, 4, "true1")
        val_img, _ = model(valid_tensors.to(device))
        save_img_tensors_as_grid(val_img, 4, "recon1")

    training_loop(
        n_epochs=200,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
    )
