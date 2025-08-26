"""
dpiTo plot and visualize the samples

Author(s): Rohit Raj (rohit.raj@ihs.uni-stuttgart.de)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.utils
from torch.utils.tensorboard import SummaryWriter

import io
from PIL import Image
from torchvision.transforms import ToTensor

from .bezier import rational_bezier_curve



# def gen_grid(latent_dim: int, 
#              points_per_axis: int, 
#              lb: int=0, 
#              rb: int=1):
#     '''Generate a grid in a d-dimensional space
#     within the range[lb, rb] for each axis '''

#     #lincoords = []
#     #for i in range(0, latent_dim):
#     #    lincoords.append(np.linspace(lb, rb, points_per_axis))
#     #coords = list(itertools.product(*lincoords))
#     #return np.array(coords)
#     lincoords = torch.linspace(lb, rb, points_per_axis)
#     mesh = torch.meshgrid(*[lincoords for _ in range(latent_dim)])
#     return torch.stack(mesh, dim=-1).view(-1, latent_dim)

# def plot_synthesized(Z,
#                      points_per_axis: int=10,
#                      gen_model: torch.nn.Module,
#                      device: torch.device,
#                      title=None):
#     ''' Plot synthesized shapes given latent space coordinates '''
#     with torch.inference_mode():
#         outputs = gen_model(Z.to(device)).cpu()

#     fig, axs = plt.subplots(points_per_axis, points_per_axis, figsize=(20, 20))

#     for i, ax in enumerate(axs.flat):
#         if outputs.dim() == 4: # Assuming [batch_size, color_channel, HEIGHT; WIDTH]
#             ax.imshow(outputs[i].permute(1, 2, 0), cmap='gray' if outputs.shape[1] == 1 else None)

        



# def plot_shape(xys, z1, z2, ax, scale, scatter,  symm_axis, **kwargs):
#     xscl = scale
#     yscl = scale

#     if scatter:
#         if 'c' not in kwargs:
#             kwargs['c'] = cm.rainbow(np.linspace(0, 1, xys.shape[0]))
        
#         ax.scatter(*zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), edgecolors='None', **kwargs)

#     else:
#         ax.plot(*zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)

#     if symm_axis == 'y':
#         plt.fill_betweenx( *zip( *[(y * yscl + z2, -x * xscl + z1, x * xscl + z1) for (x, y) in xys]), color='grey', alpha=.2)

#     elif symm_axis == 'x':
#         plt.fill_between(*zip(*[(x * xscl + z1, -y * yscl + z2, y * yscl + z2) for (x, y) in xys]), color='grey', alpha=.2)


# def plot_samples(Z, X, scale=0.8, points_per_axis=None, scatter=True, symm_axis=None, annotate=False, fname=None, **kwargs):
#     ''' Plot shapes given design space and latent space coordinates '''

#     plt.rc("font", size=12)

#     if Z is None or Z.shape[1] != 2 or points_per_axis is None:
#         N = X.shape[0]
#         points_per_axis = int(N**.5)
#         bounds = (0., 1.)
#         Z = gen_grid(2, points_per_axis, bounds[0], bounds[1])

#     scale /= points_per_axis

#     # Create a 2D plot
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111)  # three inegres as (nrows, ncols, index)

#     for (i, z) in enumerate(Z):
#         plot_shape(X[i], z[0], .3*z[1], ax, scale, scatter, symm_axis, **kwargs)
#         if annotate:
#             label = '{0}'.format(i+1)
#             ax.annotate(label, xy=(z[0], z[1]), size=10)

#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.axis('equal')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig(fname+'.svg', dpi=600)
#     plt.close()

# def plot_grid(points_per_axis: int=5,
#               gen_func: torch.nn.Module,
#               latent_dim: int=2,
#               bounds: tuple=(0., 1.),
#               scale: float=0.8,
#               scatter: bool=True,
#               symm_axis: str=None,
#               fname: str=None,
#               device: torch.device,
#               **kwargs) -> None:
#     """
#     Uniformly plots sythesized shapes in the latent space using a Pytorch generator model.

#     Args:
#     -points_per_axis (int): Number of points to sample along each axis
#     -gen_func (torch.nn.Module): Pytorch generator model
#     -latent_dim (int): Dimension of the latent space
#     -bounds (tuple): Bounds of the latent space
#     -scale (float): Scale of the plot
#     -scatter (bool): Whether to plot as scatter or line plot
#     -symm_axis (str): Axis along which symmetry is applied
#     -fname (str): File name to save the plot
#     -kwargs: Additional keyword arguments for plotting
#     """

#     # Generate a grid in the latent space
#     Z = gen_grid(latent_dim, points_per_axis, bounds[0], bounds[1])

#     # Convert the grid to a PyTorch tensor
#     Z = torch.from_numpy(Z).float()

#     # Generate shapes using the generator model
#     X = gen_func(Z).detach().numpy()

#     # Plot the synthesized shapes
#     plot_samples(Z, X, scale, points_per_axis, scatter, symm_axis, fname, **kwargs)


def log_hydFoil_images(writer: torch.utils.tensorboard,
                       generator: torch.nn.Module,
                       latent_dim: int,
                       noise_dim: int,
                       device: torch.device,
                       global_step: int,
                       num_images: int=36,
                       bound: tuple=(0, 1)) -> None:
    """
    Logs synthesized hydFoil images to TensorBoard.

    Args:
    - writer (torch.utils.tensorboard): TensorBoard writer
    - generator (torch.Module): PyTorch generator model
    - latent_dim (int): Dimension of the latent space
    - noise_dim (int): Dimension of the noise vector
    - device (torch.device): Device to run the model on
    - global_step (int): Global step for TensorBoard
    - num_images (int): Number of images to log
    """
    generator.eval() # putting generator to evaluation mode

    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Generate latent vectors
    c = torch.FloatTensor(num_images, latent_dim).uniform_(*bound).to(device)
    # Generate random noise
    z = torch.FloatTensor(num_images, noise_dim).normal_(0, 0.5).to(device)


    with torch.inference_mode():
        dp, cp, w, _, _ = generator(c, z)
        dp = dp.squeeze(1).cpu().numpy()
        cp = cp.squeeze(1).cpu().numpy()
        w = w.squeeze(1).cpu().numpy()

    t_values = np.linspace(0, 1, dp.shape[1])

    for i in range(num_images):
        row = i // grid_size
        col = i % grid_size

        # Generate points for the airfoil curves
        curve_points = np.array([
            rational_bezier_curve(t, cp[i], w[i]) for t in t_values
            ])

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
    for i in range(num_images, grid_size**2):
        row = i // grid_size
        col = i % grid_size
        fig.delaxes(axs[row, col])

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = ToTensor()(image)

    # Log image to TensorBoard
    writer.add_image('Generated HydroFoils', image, global_step)

    plt.close()