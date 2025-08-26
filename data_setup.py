'''
Contains functionality for creating PyTorch DataLoaders for 
Airfoil  data.
'''

## Create a dataLoaders
import os

from torch.utils.data import DataLoader
import torch

from hydFoil import hydFoil


# BATCH_SIZE = 256
NUM_WORKERS = os.cpu_count()

def create_dataloader(data_dir: str, 
                      file_name: str, 
                      batch_size: int=256, 
                      num_workers: int=NUM_WORKERS) -> torch.utils.data.DataLoader:
    """Create training DataLoader.

    Takes in a directory path and turns it into PyTorch datasets and 
    then turn into PyTorch DataLoaders.

    Args:
        data_dir (str): path to location of directory with numpy arrays.
        file_name (str): name of numpy array file (e.g. airfoil_interp.npy
        batch_size (int, optional): NUmber of samples per batch ineach DataLoader. Defaults to BATCH_SIZE.
        num_workers (int, optional): Number of workers per DataLoader. Defaults to NUM_WORKERS.
    

    Returns:
        A PyTorch DataLoader for training Datasets.
        Example usage:
            train_dataloader = create_dataloader(data_dir='path/to/data',
                                                 batch_size=32,
                                                 num_workers=4)
    """
    train_dataset = hydFoil(data_dir=data_dir,
                            file_name=file_name)
    
    #print("--------Writing the hydFoil samples-------------")
    #train_dataset.plot_hydFoils()
    #print("--------Done Writing the hydFoil samples-------------")
    
    #print("---------Printing data loader----------------")
    #print(train_dataset)
    #print("---------Done Printing data loader----------------")
    
    X_shape = (train_dataset[0].shape)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    return train_dataloader, X_shape  