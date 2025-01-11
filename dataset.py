import numpy as np
import torch
from torch.utils.data import Dataset

class MicroCTDataset(Dataset):
    def __init__(self, file_paths, labels, downsample_factor=2):
        self.file_paths = file_paths
        self.labels = labels
        self.downsample_factor = downsample_factor
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Read .raw file
        width = 1000  # Example dimension
        height = 1000
        depth = 500
        
        with open(self.file_paths[idx], 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            volume = data.reshape((depth, height, width))
        
        # Downsample the volume
        volume = volume[::self.downsample_factor, 
                       ::self.downsample_factor, 
                       ::self.downsample_factor]
        
        # Normalize the data
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Add channel dimension and convert to tensor
        volume = torch.FloatTensor(volume).unsqueeze(0)
        label = torch.FloatTensor([self.labels[idx]])
        
        return volume, label