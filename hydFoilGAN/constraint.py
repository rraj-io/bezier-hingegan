'''
Loss function constraints for hydFoils
'''

import torch

def closure_constraint(cp):
    # cp shape : [Batch-size, num_control_points, 2]
    first_point = cp[:, 0, :]
    last_point = cp[:, -1, :]

    return torch.mean(torch.sum((first_point - last_point)**2, dim=1))