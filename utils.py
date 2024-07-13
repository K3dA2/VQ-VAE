from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import random

def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.7002, 0.6099, 0.6036), (0.2195, 0.2234, 0.2097))  # Adjust these values if you have RGB images
    ])
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=path, transform=transform)
    
    # If num_samples is not specified, use the entire dataset
    if num_samples is None or num_samples > len(full_dataset):
        num_samples = len(full_dataset)
    print("data length: ",len(full_dataset))
    # Generate a list of indices to sample from (ensure dataset size is not exceeded)
    if shuffle:
        indices = random.sample(range(len(full_dataset)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Create a subset of the full dataset using the specified indices
    subset_dataset = Subset(full_dataset, indices)
    
    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)