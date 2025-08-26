import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os

from hydFoilGAN.gan import *
from hydFoilGAN.train import *
from utils import *
#from plot_samples import *
from data_setup import create_dataloader
from utils import training_time

import argparse
import glob

if __name__ == "__main__":

    # set the environment variable to enable expandable segments
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Arguments
    parser = argparse.ArgumentParser(description="Train a GAN model")
    parser.add_argument('mode', type=str, default='train', help='train or evalaute')
    parser.add_argument('latent', type=int, default=3, help='latent dimension')
    parser.add_argument('noise', type=int, default=10, help='noise dimension')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']


    set_seeds()

    # Hyperparameters
    LATENT_DIM = args.latent
    NOISE_DIM = args.noise
    N_POINTS = 200
    BEZIER_DEGREE = 31
    BATCH_SIZE = 64
    LR = 0.0001

    # create training dataloader
    data_loader, X_shape = create_dataloader(data_dir='hydFoil_data', 
                                             file_name='resampled_hydrofoils.npy',
                                             batch_size=BATCH_SIZE)

    #len(data_loader), type(X_shape)

    device = device_setup()
    print(f"Device is {device}")

    #generator model
    model_G = Generator(latent_dim=LATENT_DIM,
                        noise_dim=NOISE_DIM).to(device)
    #discriminator model
    model_D = Discriminator(latent_dim=LATENT_DIM).to(device)

    # Set number of epochs
    NUM_EPOCHS = 100000

    # Set loss functions
    #loss_fn = nn.BCEWithLogitsLoss()
        
    #optimizers
    optim_D = torch.optim.Adam(params=model_D.parameters(),
                               lr=LR*0.25,
                               betas=(0.0, 0.999))
    optim_G = torch.optim.Adam(params=model_G.parameters(),
                               lr=LR,
                               betas=(0.5, 0.999))
    # trainer and evaluator
    gan_trainer = GANtrainer(latent_dim=LATENT_DIM,
                            noise_dim=NOISE_DIM,
                            bound=(0.0, 1.0),
                            model_D=model_D, 
                            model_G=model_G,
                            data_loader=data_loader,
                            optim_D=optim_D,
                            optim_G=optim_G,
                            #loss_fn=loss_fn,
                            device=device)
    
    directory = './trained_gan/{}_{}'.format(LATENT_DIM, NOISE_DIM)

    # Train in args
    if args.mode == 'train':
        print("Training the model.......")
        # create training dataloader
        data_loader, X_shape = create_dataloader(data_dir='hydFoil_data',
                                                 file_name='resampled_hydrofoils.npy',
                                                 batch_size=BATCH_SIZE)

        #len(data_loader), type(X_shape)
        print("Data loader created.......")
        


        start = timer()
        Model_1_Results = gan_trainer.train(epochs=NUM_EPOCHS,
                                            directory=directory)
        end = timer()

        # Training time
        training_time(start, end, device)
        #print(f"Total training time is on {device} is {training_time(start, end, device): .3f} seconds.")
    else:
        print("Evaluating the model.......")

        # Finding the latest saved model
        model_files = glob.glob(f'{directory}/checkpoints/model_epoch_*.pth')
        if not model_files:
            raise FileNotFoundError(f"No saved models found in {directory}")
        
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Latest saved model: {latest_model}")

        # Load the saved checkpoint
        checkpoint = torch.load(latest_model, weights_only=True)

        # Load the saved model
        model_G.load_state_dict(checkpoint['model_G_state_dict'])
        model_D.load_state_dict(checkpoint['model_D_state_dict'])


    # Plot synthesized shapes:
    print("Plotting synthesized shapes.......")
    
    fname = f'{directory}/synthesized'#.format(directory)
   
    model_G.plot_synthesized_hydFoils(batch_size=BATCH_SIZE,
                                      fname=fname,
                                      epoch=NUM_EPOCHS)

    print("Synthesized shapes plotted as svg.......")
    
