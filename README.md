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

Number of Parameters: 1.6M

The model was trained on Images of size 64x64

## Reconstruction Results

Below are some examples of original images and their corresponding reconstructions by the VQ-VAE model:

![true1](https://github.com/user-attachments/assets/878207cc-9031-446c-b82e-83fcd9d5948d)

True Images

![recon1](https://github.com/user-attachments/assets/c0a98853-4158-4f82-b8a8-e165010ab41b)

Reconstructed Images

## Samples from unconditional Transfromer geneation

![jkbiugib](https://github.com/user-attachments/assets/5db43d06-7f84-465a-bf4b-a0f612ba0d23)
![ncjefhuf](https://github.com/user-attachments/assets/febb7cfb-2bfe-41ae-997a-06653617bc27)

![jcjnvjdv](https://github.com/user-attachments/assets/b7686b8f-5d18-4234-8892-7eda10c4b69b)
![jcnfnnc](https://github.com/user-attachments/assets/18ccd7e0-1912-460a-a6d0-67304a5ff98e)

![nvnovi](https://github.com/user-attachments/assets/7ce599c8-a7e9-40e9-96b3-8e62dad0de00)
![knvknvl](https://github.com/user-attachments/assets/f90e2271-57a1-4459-9b62-6dbf539015f4)

