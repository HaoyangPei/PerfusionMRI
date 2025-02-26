import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.discriminator import Discriminator
from models.spatiotemporal_generator import ST_Generator

from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
import wandb

class GAN_Module(pl.LightningModule):
    def __init__(
        self,
        in_channels=150,
        out_channels=1,
        pools: int = 4,
        chans: int = 12,
        out_map='MTT',
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pools = pools
        self.chans = chans
        self.lr = lr
        self.out_map = out_map

        # networks
        self.generator = ST_Generator(
            self.in_channels,
            self.out_channels,
            self.chans,
            self.pools)

        self.discriminator = Discriminator(in_channels+1)
        
        self.l1 = nn.SmoothL1Loss()
        self.BCELogit = nn.BCEWithLogitsLoss()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.BCELogit(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        img = batch.DCE
        
        if self.out_map=='MTT':
            label = batch.MTT[:,None,...]
        elif self.out_map=='CBF':
            label = batch.CBF[:,None,...]
        elif self.out_map=='CBV':
            label = batch.CBV[:,None,...]
        
        batch_size = img.shape[0]
        
        # train generator
        if optimizer_idx == 0:

            ones = torch.ones(batch_size,1)
            ones = ones.type_as(img)
            zeros = torch.zeros(batch_size,1)
            zeros = zeros.type_as(img)
            
            # generate images
            fake_label = self(img)
            img_fake_combine = torch.cat((img,fake_label),dim=1)
            
            fake = self.discriminator(img_fake_combine)

            l1_penalty = self.l1(label, fake_label)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(fake, ones.expand_as(fake))+100*l1_penalty
            
            self.log("l1_penalty", l1_penalty)
            self.log("g_loss", g_loss)
            
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            ones = torch.ones(batch_size,1)
            ones = ones.type_as(img)
            zeros = torch.zeros(batch_size,1)
            zeros = zeros.type_as(img)
            
            img_real_combine = torch.cat((img,label),dim=1)
            real = self.discriminator(img_real_combine)
            fake_label = self(img)
            img_fake_combine = torch.cat((img,fake_label),dim=1)
            fake = self.discriminator(img_fake_combine.detach())
            
            real_loss = self.adversarial_loss(real, ones.expand_as(real))
            fake_loss = self.adversarial_loss(fake, zeros.expand_as(fake))

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss)
            
            return d_loss
    
    def validation_step(self, batch, batch_idx):
        img = batch.DCE

        if self.out_map=='MTT':
            label = batch.MTT[:,None,...]
        elif self.out_map=='CBF':
            label = batch.CBF[:,None,...]
        elif self.out_map=='CBV':
            label = batch.CBV[:,None,...]
        
        batch_size = img.shape[0]
        
        pred_label = self.forward(img)

        l1_loss = self.l1(pred_label, label)
        
        self.log("l1_loss", l1_loss)
        
        return {
            "batch_idx": batch_idx,
            "l1_loss": l1_loss,
        }
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []