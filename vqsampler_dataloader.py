import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, file_path, file_type='txt', vocab_size=513):
        self.vocab_size = vocab_size
        self.data = self._read_file(file_path, file_type)
        self.sequences = [list(map(int, line.split())) for line in self.data]

    def _read_file(self, file_path, file_type):
        if file_type == 'txt':
            with open(file_path, 'r') as file:
                data = file.readlines()
        elif file_type == 'npy':
            data = np.load(file_path)
            data = [" ".join(map(str, line)) for line in data]
        else:
            raise ValueError("Unsupported file type. Use 'txt' or 'npy'.")
        return data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        shifted_seq = [self.vocab_size] + seq[:-1]  # Shift left and prepend 513
        return torch.tensor(shifted_seq, dtype=torch.long), torch.tensor(seq, dtype=torch.long)

def get_data_loader(file_path, file_type='txt', batch_size=32, shuffle=True, vocab_size=513):
    dataset = SequenceDataset(file_path, file_type, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example Usage
if __name__ == "__main__":
    file_path = 'indices.txt'  # Replace with your file path
    data_loader = get_data_loader(file_path, file_type='txt', batch_size=4)

    for i, (shifted_seq, original_seq) in enumerate(data_loader):
        print(f"Batch {i+1}")
        print("Shifted Sequences:\n", shifted_seq)
        print("Original Sequences:\n", original_seq)
        print(shifted_seq.shape)
        if i == 0:  # Just to print the first batch for demonstration
            break
