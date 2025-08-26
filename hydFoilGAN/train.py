
## Train step function

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, List, Tuple
from tqdm.auto import tqdm
#from typing import tuple

import os
import glob

#from . import constraint
from . import plot_samples 
#from constraint import closure_constraint

class GANtrainer:
    def __init__(self,
                 latent_dim: int,
                 noise_dim: int,
                 bound: Tuple,
                 model_D: torch.nn.Module,
                 model_G: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 #loss_fn: torch.nn.Module,
                 optim_D: torch.optim.Optimizer,
                 optim_G: torch.optim.Optimizer,
                 device: torch.device) -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent space
            noise_dim (int): Dimension of the noise vector
            model_D (torch.nn.Module): Discriminator model
            model_G (torch.nn.Module): Fake generator model
            dataloader (torch.utils.data.DataLoader): Data loader for the training data
            loss_fn (torch.nn.Module): Loss fn to train Discriminator and generator.
            optim_D (torch.optim.optimizer): Discriminator optimizer
            optim_G (torch.optim.optimizer): Generator optimizer
            device (torch.device): Device on which training to perform. Defaults to torch.device('cpu')
        """
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.bound = bound
        self.model_D = model_D
        self.model_G = model_G
        self.data_loader = data_loader
        #self.loss_fn = loss_fn
        self.optim_D = optim_D
        self.optim_G = optim_G
        self.device = device

        # Send model to device
        self.model_D.to(self.device)
        self.model_G.to(self.device)

        # Initialize loss function
        #self.loss_fn.to(self.device)

    def train_step(self,
                   writer: torch.utils.tensorboard.SummaryWriter,
                   global_step: int=1) -> tuple[float, float]:
        """
        Hinge Loss GAN train step: Trains the Discriminator (D) and generator (G) for one epoch.

        Returns:
            Tuple[float, float]: Returns the average discriminator loss and generator loss for the epoch.
        """

        # Put the critic and generator model in train mode
        self.model_D.train()
        self.model_G.train()

        # EPSILON: a small constant to avoid devision by 0
        EPSILON = 1e-8
        #lambda_gp = 10 # gradient penalty weights
        critic_iters = 1 # number of critic updates per generator update

        # Setup critic and generator losses
        disc_loss, gen_loss = 0, 0

        # Loop through dataloader data batches
        for data in self.data_loader:
            
            
            #------------------------------------------------------#
            #### New code according to beziergan
            #-------------------------------------------------------#
            # send data to target device
            real_data = data.to(self.device) # [batch_size, 200, 2]
            real_data = real_data.unsqueeze(1) # [batch_size, 1, 200, 2]
            #print(f"real_data shape is {real_data.shape}")

            # batch size
            batch_size = real_data.size(0)
            #Prepare real and fake data
            # real_labels = torch.ones(batch_size, 1).to(self.device)
            # fake_labels = torch.zeros(batch_size, 1).to(self.device)
            #print(f"fake label shape of {fake_labels.shape}")
            # y_latent is latent code: it ensures evenly explored training phase
            y_latent = torch.FloatTensor(batch_size, self.latent_dim).uniform_(*self.bound).to(self.device)
            #print(f"y_latent shape is {y_latent.shape}")
            #y_latent = y_latent.unsqueeze(1).expand(-1, 2, -1)
            #print(f"y_latent_new shape is {y_latent.shape}")
            # noise : random noise, sampled from Gaussian distri. with mean 0 and std 0.5
            noise = torch.FloatTensor(batch_size, self.noise_dim).normal_(0, 0.5).to(self.device)
            #print(f"noise shape is {noise.shape}")

            # =========================================================
            # Train Discriminator (Hinge Loss))
            # =========================================================
            for _ in range(critic_iters):
                # Generate fake airfoils
                fake_hydfoils, cp, w, ub, db = self.model_G(y_latent, noise)

                # Train discriminator
                self.optim_D.zero_grad()
                d_real, _ = self.model_D(real_data)
                #print(f"d_real shape is {d_real.shape}")
                d_fake, q_fake = self.model_D(fake_hydfoils.detach())
                #print(f"d_fake shape is {d_fake.shape}")
                #print(f"q_fake shape is {q_fake.shape}")
                #d_loss_real = self.loss_fn(d_real, real_labels)
                #d_loss_fake = self.loss_fn(d_fake, fake_labels)
                
                # Hinge Loss
                d_loss_real = torch.relu(1.0 - d_real).mean()
                d_loss_fake = torch.relu(1.0 + d_fake).mean()
                total_d_loss = d_loss_real + d_loss_fake
                
                # #total_d_loss = -torch.mean(d_real) + torch.mean(d_fake)

                # # Gradient penalty terms
                # alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
                # interpolates = alpha * real_data + (1 - alpha) * fake_hydfoils.detach()
                # interpolates.requires_grad_(True)

                # #print(f"interpolates shape is {interpolates.shape}")

                # d_interpolates, _ = self.model_D(interpolates)
                # #print(f"d_interpolates shape is {d_interpolates.shape}")

                # gradients = torch.autograd.grad(
                #     outputs=d_interpolates,
                #     inputs=interpolates,
                #     grad_outputs=torch.ones_like(d_interpolates),
                #     create_graph=True,
                #     retain_graph=True,
                #     only_inputs=True
                # )[0]

                # gradients = gradients.view(batch_size, -1)
                # grad_norm = gradients.norm(2, dim=1)
                # gradient_penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()

                # total_d_loss = d_loss + gradient_penalty
                total_d_loss.backward()
                self.optim_D.step()

            # y_latent_expanded = y_latent.unsqueeze(1).expand(-1, 2, -1)
            # q_loss = self.loss_fn(q_fake, y_latent_expanded)
            # d_loss = d_loss_real + d_loss_fake + q_loss
            # d_loss.backward()
            # self.optim_D.step()

            # =========================================================
            # Train Generator
            # =========================================================
        
            self.optim_G.zero_grad()

            # generate fake hydfoil
            fake_hydfoils, cp, w, ub, db = self.model_G(y_latent, noise)
            d_fake, q_fake = self.model_D(fake_hydfoils)

            #g_loss = -torch.mean(d_fake)#, real_labels)
            # Hinge generator loss
            g_loss = -d_fake.mean()

            # Regularization losses
            r_w_loss = torch.mean(w[:, 1:-1])
            cp_dist = torch.norm(cp[:, 1:] - cp[:, :-1], dim=-1)
            r_cp_loss = torch.mean(cp_dist)
            #r_cp_loss1 = torch.max(cp_dist)
            ends = cp[:, 0, :] - cp[:, -1, :]
            r_ends_loss = torch.mean(ends, dim=-1) + torch.clamp(-10 * ends[:, 1], min=0)
            #r_ends_loss = torch.mean(torch.sum(ends)**2, dim=1)
            #r_db_loss = torch.mean(db * torch.log(db + 1e-8))

            # Add closue penalty
            #closure_loss = constraint.closure_constraint(cp)
            #lambda_closure = 0.1 # Adjust this hyperparameter as needed

            r_loss = r_w_loss + r_ends_loss + r_cp_loss #+ 0 * r_cp_loss1 +  + 0 * r_db_loss
            

            # gaussian loss for Q (for generator update)
            q_mean = q_fake[:, 0, :]
            q_logstd = q_fake[:, 1, :]
            epsilon = (y_latent - q_mean) / (torch.exp(q_logstd) + EPSILON)
            q_loss = q_logstd + 0.5 * torch.square(epsilon)
            q_loss = torch.mean(q_loss)

            #g_total_loss = g_loss# + 10 * r_loss + q_loss
            g_total_loss = g_loss + r_loss * 5 + q_loss
            g_total_loss.mean().backward()  # .mean() reduces loss to scalar 
            self.optim_G.step()

            # Write the informations to Tensorboard summarywriter
            #writer.add_scalar('D_loss_for_real', d_loss_real.item(), global_step=global_step)
            writer.add_scalar('Critic_loss', total_d_loss.item(), global_step=global_step)
            writer.add_scalar('Generator_loss', g_loss.item(), global_step=global_step)
            writer.add_scalar('Q_loss', q_loss.item(), global_step=global_step)
            #writer.add_scalar('R_loss', r_loss.mean().item(), global_step=global_step)
            #writer.add_scalar('Closure_loss', closure_loss.item(), global_step=global_step)


            # Add the running losses to the total losses
            disc_loss += total_d_loss.item()
            gen_loss += g_total_loss.mean().item()

        return (disc_loss, gen_loss)


    # Training loop 
    def train(self,
              epochs: int=10,
              save_interval: int=5,
              directory: str='.') -> Dict[str, List[float]]:
        """Train function to train a hydFoilGAN to generate latent oriented hydFoils

        Args:
            epochs (int, optional): number of epochs over the whole training set. Defaults to NUM_EPOCHS.

        Returns:
            Dict[str, List[float]]: Returns a dictionary containing the loss values of discriminator
            and generator after each epoch.
        """
        # Create empty results dictionary
        results = {"D_loss": [],
                   "G_loss": []}
        

        latest_checkpoint = self.find_last_checkpoint(checkpoint_dir=f'{directory}/checkpoints')

        if latest_checkpoint:
            epoch, global_step = self.load_checkpoint(checkpoint_path=latest_checkpoint)
            #print(f"Previous training found with epoch_{latest_checkpoint}")
            if epoch >= epochs:
                print(f"Training already completed till epoch {latest_checkpoint}. Loading checkpoint....")
                return # skip training and proceed for evaluation
            else:
                print(f"Resuming training from epoch {epoch+1}...")
                start_epoch = epoch + 1
                
        else:
            # global step counter and beginning of training
            global_step = 0
            start_epoch = 0

        # Tensorboard Summarywriter for progress visualization 
        writer = SummaryWriter(f"{directory}/logs")
        
        # Loop through training steps for given number of epochs
        for epoch in tqdm(range(start_epoch, epochs)):
            # Increment global step
            global_step += 1
            D_loss, G_loss = self.train_step(writer=writer,
                                             global_step=global_step)

            # Append to results
            results["D_loss"].append(D_loss)
            results["G_loss"].append(G_loss)

            # Print the loss values every 5th step
            if global_step % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, D_loss: {D_loss:.4f}, G_loss: {G_loss:.4f}")
                print(f'Writing the Epoch_{epoch+1} to Writer...')
                plot_samples.log_hydFoil_images(writer=writer,
                                                generator=self.model_G,
                                                latent_dim=self.latent_dim,
                                                noise_dim=self.noise_dim,
                                                device=self.device,
                                                bound=self.bound,
                                                global_step=global_step)
                writer.flush()
            #print(f"Epoch {epoch+1}/{epochs}, D_loss: {D_loss:.4f}, G_loss: {G_loss:.4f}")

            # Save intervals for models
            save_interval = 500
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                model_path = os.path.join(directory, f"checkpoints/model_epoch_{epoch+1}.pth")
                self.save_model(epoch=epoch,
                                global_step=global_step,
                                path=model_path)
                #print(f"Model saved in path: {model_path}")

        
        # Save the final model
        #self.save_model(path=f"{directory}/model_final.pth")
        #print(f"Final model saved in path: {directory}/model_path")

        # close the writer
        writer.close()

        # return the filled results at the end of epochs
        return results


    # Save the model
    def save_model(self,
                   epoch: int,
                   global_step: int,
                   path: str) -> None:
        """Save the current model as the given file name

        Args:
            path (str, optional): path to save the model. 
        """
        # Extract the directory path from the file path
        directory = os.path.dirname(path)

        # check if the directory exists if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

        # Create the full path to the model file
        #full_path = os.path.join(save_dir, filename)

        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_D_state_dict': self.model_D.state_dict(),
            'model_G_state_dict': self.model_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
        }, path)

        print(f"Model saved to: {path}")

    # def synthesize(self,
    #                 latent: torch.Tensor,
    #                 noise: torch.Tensor=None,
    #                 return_cp: bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Synthesize shapes from the given latent code

    #     Args:
    #         latent (torch.Tensor): latent code to generate shapes
    #         noise (torch.Tensor, optional): noise to generate shapes. Defaults to None.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the generated shapes, control points and weights.
    #     """
    #     # set generator to evaluation mode
    #     self.model_G.eval()

    #     # Disable gradient computation
    #     with torch.inference_mode():
    #         if isinstance(latent, int):
    #             N = latent
    #             latent = torch.FloatTensor(N, self.latent_dim).uniform_(*self.bound).to(self.device)
    #             #print(f"Shape of the latent in 'isinstance' is not {latent.shape}")

    #             noise = torch.FloatTensor(N, self.noise_dim).normal_(0, 0,5).to(self.device)
    #             #print(f"Shape of the noise in 'isinstance' is {noise.shape}")
    #         else:
    #             N = latent.shape[0]
    #             latent = torch.FloatTensor(latent).to(self.device)
    #             #print(f"Shape of the latent in 'else' is {latent.shape}")
    #             if noise is not None:
    #                 noise = torch.FloatTensor(noise).to(self.device)
    #                 #print(f"Shape of the noise in 'else noise not none' is {noise.shape}")
    #             else:
    #                 noise = torch.zeros(N, self.noise_dim).to(self.device)
    #                 #print(f"Shape of the noise in 'else noise is none' is {noise.shape}")

    #         # Generate fake samples
    #         x_fake, p, w, _, _ = self.model_G(latent, noise)

    #         if return_cp:
    #             return x_fake, p, w
    #         else:
    #             return x_fake

    # def save_checkpoint(self,
    #                     epoch: int,
    #                     global_step: int,
    #                     path: str='checkpoints') -> None:
    #     """Save the current model as the given file name

    #     Args:
    #         path (str, optional): path to save the model.
    #     """
    #     # Extract the directory path from the file path
    #     directory = os.path.dirname(path)

    #     # Check the directory exists if not create it
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #         print(f"Created directory: {directory}")
    #     else:
    #         print(f"Directory already exists: {directory}")

    #     checkpoint = {
    #         'epoch': epoch,
    #         'global_step': global_step,
    #         'model_D_state_dict': self.model_D.state_dict(),
    #         'model_G_state_dict': self.model_G.state_dict(),
    #         'optim_D_state_dict': self.optim_D.state_dict(),
    #         'optim_G_state_dict': self.optim_G.state_dict()
    #     }
    #     torch.save(checkpoint, f'{directory}/checkpoint_epoch_{epoch+1}.pth')


    def load_checkpoint(self,
                        checkpoint_path: str) -> tuple[int, int]:
        """Load the model from the given file name

        Args:
            path (str, optional): path to load the model.
        """
        checkpoint = torch.load(checkpoint_path)

        self.model_D.load_state_dict(checkpoint['model_D_state_dict'])
        self.model_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
        self.optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

        return epoch, global_step
    
    def find_last_checkpoint(self,
                             checkpoint_dir: str) -> str:
        """Find the last checkpoint in the given directory

        Args:
            checkpoint_dir (str): path to the directory to find the last checkpoint.

        Returns:
            str: path to the last checkpoint.
        """
        
        checkpoints = glob.glob(f'{checkpoint_dir}/model_epoch_*.pth')
        if not checkpoints:
            print(f"No checkpoints found in directory: {checkpoint_dir}")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f'latest checkpoint: {latest_checkpoint}')

        return latest_checkpoint
