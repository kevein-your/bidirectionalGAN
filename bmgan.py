import torch
from torch import nn
from models.BaseModule import BaseMoDule
from networks import *

class BMGAN(BaseMoDule):
    def __init__(self, opt):
        BaseMoDule.__init__(self, opt)
        self.G_Net = networks.Dense_Generator()
        self.D_Net = networks.Patch_Discriminator()
        self.D_Net2 = networks.Patch_Discriminator()
        self.E_Net = networks.ResNet()

        self.criterion_LSGAN = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.loss_network = torch.nn.Sequential(*list(opt.vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.G_Net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_E = torch.optim.Adam(self.E_Net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_E)
        self.optimizer_D = torch.optim.Adam(self.D_Net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D)
        self.optimizer_D2 = torch.optim.Adam(self.D_Net2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D2)

    def z_guassian(self, batch_size, nz):
        z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mean_1, logvar_1 = self.E_Net.forward(input_image)
        std = logvar_1.mul(0.5).exp_()
        eps = self.z_guassian(std.size(0), std.size(1))
        z = eps.mul(std).add_(mean_1)
        return z, mean_1, logvar_1

    def backward_G(self, fake, D_Net=None):
        pred_fake = D_Net(fake)
        true_labels = t.ones_like(pred_fake)
        loss_G_GAN, _ = self.criterion_LSGAN(pred_fake, true_labels)
        return loss_G_GAN

    def backward_G_two(self):
        self.loss_z_L1 = torch.mean(torch.abs(self.mean_2 - self.z_2)) * self.opt.lambda_z
        self.loss_z_L1.backward()

    def backward_E_G(self):
        self.loss_G_GAN = self.backward_G(self.fake_data_1, self.D_Net)
        self.loss_G_GAN2 = self.backward_G(self.fake_data_random, self.D_Net2)
        self.loss_kl = torch.sum(1 + self.logvar_1 - self.mean_1.pow(2) - self.logvar_1.exp()) * (-0.5 * self.opt.lambda_kl)
        self.loss_G_L1 = self.criterion_L1(self.fake_B_encoded, self.real_B_1)
        self.loss_per = self.mse_loss(self.loss_network(fake_B_encoded), self.loss_network(real_B_1))
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_kl + 20 * self.loss_G_L1 + 8 * self.loss_per
        self.loss_G.backward(retain_graph=True)

    def backward_D(self, D_Net, real, fake):
        pred_fake = D_Net(fake.detach())
        pred_real = D_Net(real)
        true_labels = t.ones_like(pred_real)
        fake_labels = t.zeros_like(pred_fake)
        loss_D_fake, _ = self.criterion_LSGAN(pred_fake, fake_labels)
        loss_D_real, _ = self.criterion_LSGAN(pred_real, true_labels)
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def G_E_update(self):
        self.set_requires_grad([self.D_Net, self.D_Net2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_E_G()
        self.optimizer_G.step()
        self.optimizer_E.step()
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_two()
            self.optimizer_G.step()

    def D_update(self):
        self.set_requires_grad([self.D_Net, self.D_Net2], True)
        self.optimizer_D.zero_grad()
        self.loss_D, self.losses_D = self.backward_D(self.D_Net, self.real_data_1, self.fake_data_1)
        self.optimizer_D.step()
        self.optimizer_D2.zero_grad()
        self.loss_D2, self.losses_D2 = self.backward_D(self.D_Net2, self.real_data_2, self.fake_data_2)
        self.optimizer_D2.step()

    def forward(self):
        half_size = self.opt.batch_size // 2
        self.real_A_1 = self.real_A[0:half_size]
        self.real_B_1 = self.real_B[0:half_size]
        self.real_B_2 = self.real_B[half_size:]
        self.z_1, self.mean_1, self.logvar_1 = self.encode(self.real_B_1)
        self.z_2 = self.z_guassian(self.real_A_1.size(0), self.opt.nz)
        self.fake_B_1= self.G_Net(self.real_A_1, self.z_1)
        self.fake_B_2 = self.G_Net(self.real_A_1, self.z_2)
        self.fake_data_1 = torch.cat([self.real_A_1, self.fake_B_encoded], 1)
        self.real_data_1 = torch.cat([self.real_A_1, self.real_B_1], 1)
        self.fake_data_2 = torch.cat([self.real_A_1, self.fake_B_2], 1)
        self.real_data_2 = torch.cat([self.real_A[half_size:], self.real_B_2], 1)
        self.mean_2, logvar_2 = self.E_Net(self.fake_B_2)

    def optimize_parameters(self):
        self.forward()
        self.G_E_update()
        self.D_update()
