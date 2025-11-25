# Bezier-HingeGAN

Bezier-HingeGan is an extension of the work of Bezier-GAN work, earlier presented by Wei Chen. The following code is written with PyTorch. It generates stable and diverse aerodynamic shapes with hinge generative loss.

## Overview

Bezier-HingeGAN is a generative adversarial network model designed for *stable and diverse aero-hydrodynamic blade and airfoil shape generation*. Unlike typical GANs that generates point cloud or pixel data, Bezier-HingeGAN generates **Bezier control points and weights**, which are then rendered into smooth parametric blade/airfoil geometries.

This makes the method idea for:
 - Aerodynamic and hydropower turbine blade design
 - Generative design for engineering CAD geometries
 - Surrogate-based optimizatioh in latent space
 - Physics-informed or CFD-coupled workflows

The model improves training stability and diversity using a **hinge-loss based discriminator**.

---

## Model Architecture

![Alt text](./results/beziergan-arch.svg)


--- 

## 


