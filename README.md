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
Number of Parameters: 31 million
The model was trained for 100 epochs, with the training loss plotted over time as shown below:

![c28bffa5-a74e-4d68-a9e2-45e6716baeba](https://github.com/user-attachments/assets/8c029c9b-fcd0-4229-bdf5-5a8b4cb4b2ce)

## Reconstruction Results

Below are some examples of original images and their corresponding reconstructions by the VQ-VAE model:

![true1 copy](https://github.com/user-attachments/assets/c6770715-8256-4dbe-be3b-0c06378ae760)

True Images

![recon1 copy](https://github.com/user-attachments/assets/7dd93aac-f6ae-4fb7-b8bd-bbb95aa5945c)

Reconstructed Images

