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
#from model2 import VQVAE
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid
import uuid
from torch.utils.tensorboard import SummaryWriter


def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, valid_loader,
                  max_grad_norm=1.0, epoch_start=0, save_img=True, show_img=False,
                ema_alpha=0.99):

    model.train()
    writer = SummaryWriter(log_dir='./logs')
    ema_loss = None
    scheduler = None

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

            
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            if show_img:
                pred_images = model.inference(1, 16, 16)
                plt.imshow(np.transpose(pred_images[-1].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()
            if save_img:
                pred_images = model.inference(1, 16, 16)
                pred_images = np.transpose(pred_images[-1].cpu().detach().numpy(), (1, 2, 0))
                random_filename = str(uuid.uuid4()) + '.png'

                # Specify the directory where you want to save the image
                save_directory = ""

                # Create the full path including the directory and filename
                full_path = os.path.join(save_directory, random_filename)
                # Save the image with the random filename
                plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
                with torch.no_grad():
                    for valid_tensors, _ in valid_loader:
                        break

                    save_img_tensors_as_grid(valid_tensors, 4, "true1")
                    val_img, _ = model(valid_tensors.to(device))
                    save_img_tensors_as_grid(val_img, 4, "recon1")

            model_path = os.path.join('/Users/ayanfe/Documents/Code/VQ-VAE/weights', 'waifu-vqvae_epoch.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)



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
    
    
    print(model.inference(1, 16, 16).shape)
    with torch.no_grad():
        for valid_tensors, _ in val_loader:
            break

        save_img_tensors_as_grid(valid_tensors, 4, "true1")
        val_img, _ = model(valid_tensors.to(device))
        save_img_tensors_as_grid(val_img, 4, "recon1")

    training_loop(
        n_epochs=100,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
    )
