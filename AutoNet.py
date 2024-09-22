import lightning
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import time

class AutoNet(lightning.LightningModule):
    def __init__(self, Generator, Discriminator, nz, nc, ngf, ndf, lr, beta1):
        super(AutoNet, self).__init__()
        self.automatic_optimization = False

        # * Define the generator and discriminator
        self.G = Generator(nz, ngf, nc)
        self.D = Discriminator(nc, ndf)
        
        # * Define the hyperparameters & device & labels
        self.lr = lr
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.beta = beta1

        self.real_label = 1
        self.fake_label = 0

        # * initialize the weights
        self.G.apply(self.weight_init)
        self.D.apply(self.weight_init)

        self.criterion = nn.BCELoss()

        # * define logger
        if not os.path.exists(f'./logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}'):
            os.makedirs(f'./logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}')
        logging.basicConfig(filename=f'./logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}/training.log', level=logging.INFO)


    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # * update the state_dict because of two models
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        })
        return state_dict
    
    # def load_state_dict(self, state_dict, strict=True):
    #     self.G.load_state_dict(state_dict.pop('G'))
    #     self.D.load_state_dict(state_dict.pop('D'))
    #     super().load_state_dict(state_dict, strict)
    #     return self

    def training_step(self, batch, batch_idx):
        
        opt_G, opt_D = self.optimizers()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        real, _ = batch
        b_size = real.size(0)
        label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
        output = self.D(real).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
        fake = self.G(noise)
        label.fill_(0)
        output = self.D(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        opt_D.step()
        opt_D.zero_grad()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        label.fill_(self.real_label)
        output = self.D(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        opt_G.step()
        opt_G.zero_grad()

        # * saving & logging
        self.errG = errG
        self.errD = errD
        self.D_x = D_x
        self.D_G_z1 = D_G_z1
        self.D_G_z2 = D_G_z2

        self.log('errD', errD)
        self.log('D_x', D_x)
        self.log('errG', errG)
        self.logger.experiment.add_scalars('D(G(z))', {'D(G(z))_1': D_G_z1, 'D(G(z))_2': D_G_z2}, self.current_epoch)

    def on_train_epoch_end(self):
        logging.info(
            f'Epoch: {self.current_epoch}',
            f'Loss-D: {self.errD.item():.4f}',
            f'Loss-G: {self.errG.item():.4f}',
            f'D(x): {self.D_x:.4f}',
            f'D(G(z)): [{self.D_G_z1:.4f}/{self.D_G_z2:.4f}]',
        )

        # *save the model
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

        checkpoint_path = f'./checkpoints/epoch_{self.current_epoch}.ckpt'
        torch.save(self.state_dict(), checkpoint_path)
        



    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta, 0.999))

        return opt_G, opt_D
