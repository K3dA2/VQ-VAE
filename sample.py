import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformer_model import VQSampler, Config
from model import VQVAE
import os
import numpy as np

def sample(vq_vae_model, sampler_model, seq=torch.tensor([[512]]), seq_len=257, display=True, save_dir='Sampled'):
    sampler_model.to(seq.device)
    vq_vae_model.to(seq.device)
    sampler_model.eval()
    vq_vae_model.eval()
    while seq.size(1) < seq_len:
        with torch.no_grad():
            logits, _ = sampler_model(seq)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            topk_prob, topk_indices = torch.topk(prob, 30, dim=-1)
            idx = torch.multinomial(topk_prob, 1)
            seq_col = torch.gather(topk_indices, -1, idx)
            seq = torch.cat((seq, seq_col), dim=1)
    print(seq)
    seq = seq[:,1:]
    #seq = torch.tensor([[7, 129, 149, 229, 429, 81, 229, 351, 294, 95, 343, 355, 49, 34, 40, 272, 173, 439, 485, 1, 263, 397, 251, 191, 375, 332, 156, 508, 223, 27, 387, 90, 403, 449, 111, 459, 485, 51, 124, 111, 355, 397, 439, 485, 167, 315, 282, 494, 485, 485, 366, 159, 485, 85, 95, 89, 159, 354, 159, 218, 319, 272, 36, 427, 360, 478, 439, 366, 85, 439, 485, 2, 161, 485, 149, 149, 403, 41, 176, 176, 89, 485, 474, 485, 407, 420, 72, 368, 95, 85, 448, 474, 485, 92, 384, 316, 485, 411, 477, 485, 124, 87, 276, 296, 109, 379, 456, 485, 191, 262, 241, 165, 206, 485, 356, 485, 229, 37, 312, 482, 492, 169, 107, 485, 350, 285, 318, 433, 507, 89, 325, 437, 485, 394, 238, 505, 492, 241, 68, 485, 336, 413, 353, 258, 332, 166, 485, 273, 184, 472, 143, 180, 492, 118, 485, 409, 300, 454, 92, 143, 485, 37, 485, 332, 382, 97, 56, 307, 56, 241, 124, 435, 346, 103, 192, 428, 354, 126, 485, 210, 336, 509, 371, 387, 48, 92, 414, 446, 430, 430, 353, 126, 68, 343, 336, 464, 431, 48, 54, 223, 382, 16, 430, 137, 233, 328, 103, 14, 198, 182, 43, 505, 174, 40, 253, 96, 318, 52, 71, 176, 430, 318, 108, 252, 6, 174, 177, 496, 9, 270, 256, 33, 96, 401, 174, 220, 285, 310, 318, 274, 443, 180, 505, 177, 401, 467, 263, 182, 49, 238, 52, 48, 275, 429, 336, 119]])
    print(seq.shape)
    with torch.no_grad():
        img = vq_vae_model.decode(seq)

    # Assuming img is a tensor in the format (C, H, W) and needs to be converted to (H, W, C) for plotting
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    mean=[0.7002, 0.6099, 0.6036]
    std=[0.2195, 0.2234, 0.2097]
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    if display:
        # Display the image
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        # Save the image
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, 'sampled_image.png')
        plt.imsave(img_path, img)

if __name__ == "__main__":
    vq_model_path = '/Users/ayanfe/Documents/Code/simple vq-vae/weights/waifu-vqvae_epoch.pth'
    sampler_model_path = '/Users/ayanfe/Documents/Code/simple vq-vae/weights/vqvae-sampler.pth'

    config = Config(num_layers = 8, emb_dim = 72, num_heads = 12, seq_len=256)
    sampler = VQSampler(config)
    vqvae = VQVAE(latent_dim = 12, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64)  

    checkpoint = torch.load(vq_model_path)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(sampler_model_path)
    sampler.load_state_dict(checkpoint['model_state_dict'])

    sample(
        vq_vae_model=vqvae,
        sampler_model=sampler,
        display=True,  
        save_dir='Sampled'  
    )
