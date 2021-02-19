import torch
from models.BaseModule import BaseMoDule
from networks import *

class BMGAN(BaseMoDule):
    def __init__(self):
        BaseMoDule.__init__(self,opt)
        self.model_names = ['G']
        self.netG = networks.Dense_Generator()
        self.model_names += ['D']
        self.netD = networks.Patch_Discriminator()
        self.model_names += ['D2']
        self.netD2 = networks.Patch_Discriminator()
        self.model_names += ['E']
        self.netE = networks.ResNet()

        self.criterionGAN = nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionZ = torch.nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.vgg = vgg(pretrained=True)
        self.loss_network = nn.Sequential(*list(self.vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_E)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D)
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D2)

    def get_z_random(self, batch_size, nz):
        z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def forward(self):
        half_size = self.opt.batch_size // 2
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)
        self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
        self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
        self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
        self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        self.mu2, logvar2 = self.netE(self.fake_B_random)

    def backward_D(self, netD, real, fake):
        pred_fake = netD(fake.detach())
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None):
        pred_fake = netD(fake)
        loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        return loss_G_GAN

    def backward_EG(self):
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD)
        self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2)
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        self.loss_per = self.mse_loss(self.loss_network(fake_B_encoded), self.loss_network(real_B_encoded))
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_kl + 20 * self.loss_G_L1 + 8 * self.loss_per
        self.loss_G.backward(retain_graph=True)

    def backward_G_alone(self):
        self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
        self.loss_z_L1.backward()

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def update_G_and_E(self):
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B