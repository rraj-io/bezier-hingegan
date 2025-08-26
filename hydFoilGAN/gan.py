### Discriminator Architecture same as given by Bezier GAN paper


import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from utils import device_setup

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .bezier import *
import io
import os

class Discriminator(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            input_shape: int=1,
            depth: int=64,
            kernel_size: int | Tuple=(4,2),
            stride: int | Tuple=1,
            dropout: float=0.4
        ) -> None:
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.depth = depth
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        #self.padding = padding
        self.dropout = dropout
        
        #self.conv1 = self._conv_block(self.input_shape, self.depth*1, self.kernel_size, self.stride, self.padding, self.dropout)

        self.disc = nn.Sequential(
            self._conv_block(self.input_shape, self.depth*1, self.kernel_size, self.stride, self.dropout),
            self._conv_block(self.depth*1, self.depth*2, self.kernel_size, self.stride, self.dropout),
            self._conv_block(self.depth*2, self.depth*4, self.kernel_size, self.stride, self.dropout),
            self._conv_block(self.depth*4, self.depth*8, self.kernel_size, self.stride, self.dropout),
            #self._conv_block(self.depth*8, self.depth*16, self.kernel_size, self.stride, self.dropout),
            #self._conv_block(self.depth*16, self.depth*32, self.kernel_size, self.stride, self.dropout)
        )
        
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.depth*8 * 2 * 200, 1024),   # changed size from 32->8
            nn.BatchNorm1d(1024, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.linear = nn.Linear(1024, 1)

        self.q = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.q_mean = nn.Linear(128, self.latent_dim)

        self.q_logstd = nn.Linear(128, self.latent_dim)  


    def forward(self, x):
        #x = x.permute(0, 2, 1).unsqueeze(1)  # Reshape to [256, 1, 2, 192]
        #print(f"size of x is {x.shape}")
        
        x = self.flatten(self.disc(x))
        #print(f"size of flattened X in line 75 is {x.shape}")
        
        d = self.linear(x)
        #print(f"size of d in line 78 is {d.shape}")

        q = self.q(x)
        #print(f"size of q is {q.shape}")
        q_mean = self.q_mean(q)
        #print(f"size of q_mean is {q_mean.shape}")
        q_logstd = torch.clamp(self.q_logstd(q), min=-16)
        ##print(f"size of q_logstd is {q_logstd.shape}")

        # Reshape to [batch_size x 1 x latent_dim]
        q_mean = q_mean.unsqueeze(1)
        #print(f"size of q_mean after unsqueeze is {q_mean.shape}")
        q_logstd = q_logstd.unsqueeze(1)
        #print(f"size of q_logstd after unsqueeze is {q_logstd.shape}")

        q = torch.cat([q_mean, q_logstd], dim=1) # [batch_size, 2, latent_dim]
        #print(f"size of q is {q.shape}")
        return d, q

    
    def _conv_block(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               DROPOUT
        ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same', bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(DROPOUT)
        )
### Generator Architecture same as given by Bezier GAN paper
#device = device_setup()
#EPSILON = torch.tensor(1e-8, device=device)

class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 noise_dim: int,
                 n_points: int=200,
                 bezier_degree: int=31,
                 bounds: int | Tuple=(0.0, 1.0),
                 kernel_size: int | Tuple=(4, 3),
                 stride: int | Tuple=(2, 1),
                 padding: int | Tuple=(1, 1),
                 device: torch.device = device_setup()
                 ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.n_points = n_points
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.EPSILON = torch.tensor(1e-8, device=self.device)
        
        #self.input_dim   = latent_dim + noise_dim if noise_dim > 0 else latent_dim

        #self.depth_cpw = 32 * 8
        #self.dim_cpw = int((self.bezier_degree + 1) / 8)
        self.input_dim = self.latent_dim + self.noise_dim if self.noise_dim > 0 else self.latent_dim
        self.depth_cpw = 32 * 8
        self.dim_cpw = int((bezier_degree + 2) / 8)

        self.gen1 = nn.Sequential(
            self._linear_block(self.input_dim, 1024),
            self._linear_block(1024, self.dim_cpw*3*self.depth_cpw)
        )
            #nn.view(-1, (self.dim_cpw, 3, self.depth_cpw)),
            #lambda x: self._custom_reshape(x),
        self.gen2 = nn.Sequential(
            self._Tconv_block(self.depth_cpw, int(self.depth_cpw/2), self.kernel_size, self.stride, self.padding),
            self._Tconv_block(int(self.depth_cpw/2), int(self.depth_cpw/4), self.kernel_size, self.stride, self.padding),
            self._Tconv_block(int(self.depth_cpw/4), int(self.depth_cpw/8), self.kernel_size, self.stride, self.padding),
        )

        self.cp_conv2d = nn.Conv2d(int(self.depth_cpw/8), 1, (1, 2), padding=(0,0))
        self.w_conv2d = nn.Conv2d(int(self.depth_cpw/8), 1, (1, 3), padding=(0,0))

        self.db_gen = nn.Sequential(
            self._linear_block(self.input_dim, 1024),
            self._linear_block(1024, 256),
            #nn.Linear(256, self.X_shape[0]-1),
            nn.Linear(256, self.n_points-1),
        )

    def forward(self, c, z=None):
        if self.noise_dim == 0 or z is None:
            cz = c
        else:
            cz = torch.cat([c, z], dim=1)

        #print(f"CZ is {cz.shape}")

        # Control points and weight generation together
        cpw = self.gen1(cz)
        #print(f"1. cpw is {cpw.shape}")

        cpw = cpw.view(-1, self.depth_cpw, self.dim_cpw, 3) # -> [batch_size, depth_cpw, dim_cpw, 3]
        #print(f"2. cpw is {cpw.shape}")

        cpw = self.gen2(cpw)  
        #print(f"3. cpw is {cpw.shape}")

        # Control points
        cp = nn.functional.tanh(self.cp_conv2d(cpw)).squeeze(1) #-> [batch_size, bezier_degree+1, 2]
        #print(f"cp is {cp.shape}")

        # Ensures closure by making the last control point equal to the first
        cp = torch.cat([cp, cp[:, :1, :]], dim=1) # Append first point as last point
        #print(f"cp is {cp.shape}")

        # weights
        w = nn.functional.softmax(self.w_conv2d(cpw), dim=-1).squeeze(1) #-> [batch_size, bezier_degree+1, 1]
        #print(f"w is {w.shape}")

        # Added change: ensures weight for teh last control point is equal to first
        w = torch.cat([w, w[:, :1]], dim=1) # Append first weight as last weight
        #print(f"w is {w.shape}")

        # Parameters at data points
        db = nn.functional.softmax(self.db_gen(cz), dim=1) # -> [batch_size, n_points-1]
        #print(f"db is {db.shape}")

        ub = nn.functional.pad(db, (1, 0), value=0)
        ub = torch.cumsum(ub, dim=1)
        ub = torch.clamp(ub, max=1).unsqueeze(-1)
        #print(f"ub is {ub.shape}")  # -> [batch_size, n_points-1, 1]

        # Bezier layer
        # Compute vaues of basis functions at data points
        num_control_points = self.bezier_degree + 2
        #print(f"num_control_points is {num_control_points}")
        lbs = ub.repeat(1, 1, num_control_points)
        #print(f"lbs is {lbs.shape}")
        pw1 = torch.arange(0, num_control_points, dtype=torch.float32, device=lbs.device).view(1, 1, -1) # -> [1, 1, n_control_points]
        #print(f"pw1 is {pw1.shape}")    
        pw2 = torch.flip(pw1, dims=[-1])
        #print(f"pw2 is {pw2.shape}")

        # print("lbs device:", lbs.device)
        # print("pw1 device:", pw1.device)
        # print("pw2 device:", pw2.device)
        # print("self.EPSILON device:", self.EPSILON.device)
        lbs = pw1 * torch.log(lbs + self.EPSILON) + pw2 * torch.log(1 - lbs + self.EPSILON) #-> [batch_size, n_points, n_control_points]
        #print(f"lbs is {lbs.shape}") 
        lc = torch.lgamma(pw1 + 1) + torch.lgamma(pw2 + 1)
        lc = torch.lgamma(torch.tensor(num_control_points, dtype=torch.float32)) - lc # [1, 1, n_control_points]
        #print(f"lc is {lc.shape}")
        lbs = lbs + lc # [batch_size, n_data_points, n_control_points]
        #print(f"lbs is {lbs.shape}")
        bs = torch.exp(lbs)
        #print(f"bs is {bs.shape}")

        # Compute data points
        cp_w = cp * w
        #print(f"cp_w is {cp_w.shape}")
        dp = torch.matmul(bs, cp_w)#.transpose(1, 2)) # [batch_size, n_points, 2]
        #print(f"dp is {dp.shape}")
        bs_w = torch.matmul(bs, w)#.transpose(1, 2)) # [batch_size, n_points, 1]
        #print(f"bs_w is {bs_w.shape}")
        dp /= (bs_w + self.EPSILON)  # [batch_size, n_points, 2]
        #print(f"dp is {dp.shape}")
        dp = dp.unsqueeze(1)   # [batch_size,1 , n_points, 2]
        #print(f"dp is {dp.shape}")

        #print(f"dp is {dp.shape}")
        #print(f"cp is {cp.shape}")
        #print(f"w is {w.shape}")

        return dp, cp, w, ub, db


    def _Tconv_block(
            self,
            input, 
            depth,
            kernel_size,
            stride,
            padding
        ):
        return nn.Sequential(
            nn.ConvTranspose2d(input, depth, kernel_size, stride, padding),
            nn.BatchNorm2d(depth, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def _linear_block(self,
                      input,
                      depth):
        return nn.Sequential(
            nn.Linear(input,depth),
            nn.BatchNorm1d(depth, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)            
        )
    
    # def _custom_reshape(
    #         x#,
    #         #dim_cpw,
    #         #depth_cpw,
    # ):
    #     return x.view(-1, self.dim_cpw, 3, self.depth_cpw)
    def plot_synthesized_hydFoils(self,
                                  batch_size,
                                  fname,
                                  epoch,
                                  bound=(0., 1.),
                                  num_samples=36):
        
        self.eval() # set generator to evaluation mode
        
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

        #for i in range(num_samples):
        #    row = i // grid_size
        #    col = i % grid_size

        # Generate a random latent vector
        c = torch.FloatTensor(batch_size, self.latent_dim).uniform_(*bound).to(self.device)
        z = torch.FloatTensor(batch_size, self.noise_dim).normal_(0, 0.5).to(self.device)

        with torch.inference_mode():
            dp, cp, w, _, _ = self.forward(c, z)
            dp = dp.squeeze(1).cpu().numpy()
            cp = cp.squeeze(1).cpu().numpy()
            w = w.squeeze(1).cpu().numpy()

            #print(f"Shape of dp in inf_m is {dp.shape}")
            #print(f"Shape of cp in inf_m is {cp.shape}")
            #print(f"Shape of w in inf_m is {w.shape}"

        # randomly samples indices
        sample_indices = np.random.choice(batch_size, num_samples, replace=True)

        for i, idx in enumerate(sample_indices):
            row = i // grid_size
            col = i % grid_size

            # Generate points for the airfoil curves
            t_values = np.linspace(0, 1, self.n_points)
            #upper_surface = np.array([rational_bezier_curve(ti, dp[i, :96], w[i]) for ti in t])
            #lower_surface = np.array([rational_bezier_curve(ti, dp[i, 96:], w[i]) for ti in t])
            curve_points = np.array([rational_bezier_curve(t, dp[idx, :, :], w[idx, :, :]) for t in t_values])

            # appending first point to the end point
            curve_points = np.vstack([curve_points, curve_points[0]])

            # Plot the airfoils
            axs[row, col].plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=0.8)
            #axs[row, col].plot(dp[i, :, 0], dp[i, :, 1], 'ro-')#, linewidth=0.8)
            axs[row, col].axis('off')
            axs[row, col].set_aspect('equal')
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

        # remove unused subplots
        for i in range(num_samples, grid_size**2):
            row = i // grid_size
            col = i % grid_size
            fig.delaxes(axs[row, col])

        plt.tight_layout()

        # Ensure that the directory exists
        os.makedirs(fname, exist_ok=True)

        # create the full path for the output file
        output_filename = os.path.join(fname, f'synthesized_hydFoils_{epoch}.svg')

        # Save the figure to a BytersIP object
        buf = io.BytesIO()
        plt.savefig(buf, format='svg', dpi=600)
        buf.seek(0)

        # Save the figure to a file
        with open(output_filename, 'wb') as f:
            f.write(buf.getvalue())

        plt.close()
        print(f'Figure saved to {output_filename}')

    # def generate_hydrofoil_pointclouds(self, N_samples, bound=(0., 1.)):
    #     self.eval()

    #     batch_size = 512  # or whatever your GPU memory allows
    #     n_batches = int(np.ceil(N_samples / batch_size))
    #     all_pointclouds = []

    #     for _ in range(n_batches):
    #         current_batch = min(batch_size, N_samples - len(all_pointclouds))

    #         c = torch.FloatTensor(current_batch, self.latent_dim).uniform_(*bound).to(self.device)
    #         z = torch.FloatTensor(current_batch, self.noise_dim).normal_(0, 0.5).to(self.device)

    #         with torch.inference_mode():
    #             dp, cp, w, _, _ = self.forward(c, z)
    #             dp = dp.squeeze(1).cpu().numpy()  # (batch, ctrl_pts, 2)
    #             w = w.squeeze(1).cpu().numpy()    # (batch, ctrl_pts, 2)

    #         t_values = np.linspace(0, 1, self.n_points)

    #         for i in range(current_batch):
    #             curve_points = np.array([rational_bezier_curve(t, dp[i, :, :], w[i, :, :]) for t in t_values])
    #             # Optionally, close the airfoil by appending first point
    #             curve_points = np.vstack([curve_points, curve_points[0]])
    #             all_pointclouds.append(curve_points)

    #     return np.array(all_pointclouds)  # shape: (N_samples, n_points+1, 2)
    
    # def generate_hydrofoil_pointclouds(self, 
    #                                    N_samples,
    #                                    batch_size, 
    #                                    bound=(0., 1.)):
    #     self.eval()

    #     all_pointclouds = []
        
    #     num_batches = int(np.ceil(N_samples / batch_size))

    #     with torch.inference_mode():
    #         for batch_idx in range(num_batches):
    #             # Generate random latent vectors
    #             current_batch_size = min(batch_size, N_samples - batch_idx * batch_size)

    #             c = torch.FloatTensor(current_batch_size, self.latent_dim).uniform_(*bound).to(self.device)
    #             z = torch.FloatTensor(current_batch_size, self.noise_dim).normal_(0, 0.5).to(self.device)

    #             dp, cp, w, _, _ = self.forward(c, z)
    #             dp = dp.squeeze(1).cpu().numpy()  # (ctrl_pts, 2)
    #             cp = cp.squeeze(1).cpu().numpy()  # (ctrl_pts, 2)
    #             w  = w.squeeze(1).cpu().numpy()   # (ctrl_pts, 1)
    #             #n_points = cp.shape[1]
    #             t_values = np.linspace(0, 1, self.n_points)

    #             for idx in range(current_batch_size):
    #                 curve_points = np.array([rational_bezier_curve(t, cp[idx], w[idx]) for t in t_values])
    #                 #curve_points = np.vstack([curve_points, curve_points[0]])
    #                 all_pointclouds.append(curve_points)

    #     return np.array(all_pointclouds)  # shape: (N_samples, n_points+1, 2)

