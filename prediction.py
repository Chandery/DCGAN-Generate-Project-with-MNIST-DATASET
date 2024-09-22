import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import lightning
import os


def set_hyperparameters():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Root directory for dataset
    dataroot = "data/mnist"
    # Number of workers for dataloader
    workers = 10
    # Batch size during training
    batch_size = 100
    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size = 64
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 10
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    return dataroot, workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu

def predict():
    dataroot, workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu = set_hyperparameters()

    from models.Generator import G
    from models.Discriminator import D
    
    # Create the models
    G_model = G(nz, ngf, nc)
    D_model = D(nc, ndf)

    checkpoint = torch.load("checkpoints/epoch_9.ckpt")

    # Load the state of the models from the checkpoint
    G_model.load_state_dict(checkpoint['G'])
    D_model.load_state_dict(checkpoint['D'])

    # Put the models in evaluation mode
    G_model.eval()
    D_model.eval()

    fixed_noise = torch.randn(100, nz, 1, 1)

    # Generate fake images
    fake_images = G_model(fixed_noise)

    # Save the images
    utils.save_image(fake_images, "fake_images.png", nrow=10, normalize=True)

    


if __name__ == '__main__':
    predict()