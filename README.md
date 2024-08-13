# VQ-VAE
 Overview

Vector Quantized Variational Autoencoders (VQ-VAE) are a type of generative model that combines the power of variational autoencoders with discrete latent variables. This allows for more efficient encoding of information and can lead to better performance on various tasks, such as image reconstruction and generation.

## Model Architecture

The architecture of the VQ-VAE model includes:

An encoder network that maps the input data to a discrete latent space.
A vector quantization layer that replaces continuous latent variables with discrete codes.
A decoder network that reconstructs the input data from the discrete codes.
Training Details

Hardware: M1 Max MacBook Pro

Epochs: 100

Number of Parameters: 600k

## Reconstruction Results

Below are some examples of original images and their corresponding reconstructions by the VQ-VAE model:

![true1](https://github.com/user-attachments/assets/4ced3f78-7c2d-4920-a67c-aa29a9cd6c83)

True Images

![recon1](https://github.com/user-attachments/assets/3b62decb-4512-4214-a67c-bf231bcead32)

Reconstructed Images

![xtrue](https://github.com/user-attachments/assets/af07f3f9-6b38-41d5-8267-05bd1be64fb8)

True Image

![xrecon](https://github.com/user-attachments/assets/66972d5b-3696-4a61-9e4d-b2ee13236ae8)

Reconstructed Image
