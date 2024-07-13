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
from utils import get_data_loader,count_parameters
import uuid


def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader,\
                   max_grad_norm=1.0, epoch_start = 0,\
                    save_img = False, show_img = True):
    model.train()
    for epoch in range(epoch_start,n_epochs):
        loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)

            
            outputs,c_loss = model(imgs)
            loss = loss_fn(outputs, imgs) + c_loss
            
            optimizer.zero_grad()
            loss.backward()
            #utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename
        model_filename = f'waifu-vqvae_epoch_{epoch}.pth'
        model_path = os.path.join('/Users/ayanfe/Documents/Code/VQ-VAE/weights', model_filename)
        
        with open("waifu-vqvae_epoch-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        if epoch % 10 == 0:
            if show_img:
                pred_images = model.inference(1, 14, 14)
                plt.imshow(np.transpose(pred_images[-1].cpu().detach().numpy(), (1, 2, 0)))
                plt.show()
            if save_img:
                pred_images = model.inference(1, 14, 14)
                pred_images = np.transpose(pred_images[-1].cpu().detach().numpy(), (1, 2, 0))
                random_filename = str(uuid.uuid4()) + '.png'

                # Specify the directory where you want to save the image
                save_directory = path

                # Create the full path including the directory and filename
                full_path = os.path.join(save_directory, random_filename)
                # Save the image with the random filename
                plt.savefig(full_path, bbox_inches='tight', pad_inches=0)

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)


if __name__ == "__main__":
    path = '/Users/ayanfe/Documents/Datasets/Waifus'
    #model_path = '/Users/ayanfe/Documents/Code/Diffusion-Model/weights/waifu-diffusion-cts_epoch_80.pth'
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model = VQVAE()  # Assuming Unet is correctly imported and defined
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #loss_fn = nn.L1Loss().to(device)
    loss_fn = nn.MSELoss().to(device)
    print(count_parameters(model))
    data_loader = get_data_loader(path, batch_size=16,num_samples=1_000)

    # Optionally load model weights if needed
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        epoch_start= 0,
    )
    
