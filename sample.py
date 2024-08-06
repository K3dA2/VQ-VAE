import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformer_model import VQSampler, Config
from model import VQVAE
import os
import numpy as np

def sample(vq_vae_model, sampler_model, seq=torch.tensor([[299]]), seq_len=256, display=True, save_dir='Sampled'):
    sampler_model.to(seq.device)
    vq_vae_model.to(seq.device)
    sampler_model.eval()
    vq_vae_model.eval()
    while seq.size(1) < seq_len:
        with torch.no_grad():
            logits, _ = sampler_model(seq)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            topk_prob, topk_indices = torch.topk(prob, 4, dim=-1)
            idx = torch.multinomial(topk_prob, 1)
            seq_col = torch.gather(topk_indices, -1, idx)
            seq = torch.cat((seq, seq_col), dim=1)

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
    vq_model_path = ''
    sampler_model_path = ''

    config = Config(num_layers=4, emb_dim=32, num_heads=4, seq_len=256)
    sampler = VQSampler(config)
    vqvae = VQVAE(latent_dim=1, num_embeddings=512, beta=0.25, use_ema=False)  # Assuming VQVAE is correctly imported and defined

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
