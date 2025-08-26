# Create a class for custom dataset:

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

class hydFoil(Dataset):
    def __init__(self, 
                 data_dir: str,
                 file_name: str) -> None:
        """Initialize the dataset with a NumPy array.

        Args:
            data_dir (str): Dirctory containing the input data as an ndarray
            file_name (str): Nmae of the .npy file containing the data
        """
        file_path = os.path.join(data_dir, file_name) # Full path to the data file
        self.data = np.load(file_path) # load the data directly here 

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The sample at the specified index.
        """
    
        return torch.tensor(self.data[idx], dtype=torch.float32)
    
    def plot_hydFoils(self, scale=0.8, num_samples=36) -> None:
        """
        Plot random samples in the dataset
        """
            
        # Generate random indices
        indices = np.random.choice(len(self), num_samples, replace=False)

        # calcualate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        # Create a figure with subplots
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(5, 5))

        for i, idx in enumerate(indices):
            row = i // grid_size
            col = i % grid_size

            hydfoil = self[idx].numpy()

            x, y = hydfoil[:, 0], hydfoil[:, 1]

            axs[row, col].plot(x, y, '-b', linewidth=0.8)
            axs[row, col].scatter(x, y, s=1, c='r')
            axs[row, col].set_aspect('equal')
            axs[row, col].axis('off')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

        plt.tight_layout()
        #plt.show()
        plt.savefig("./hydFoil_data/hydfoil_samples.svg", dpi=600)
        plt.close()
