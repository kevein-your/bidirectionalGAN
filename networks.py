import torch as t
from torch import nn
from models.BasicModule import BasicMoDule
from torch.nn import functional as F
import functools

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool3d(kernel_size=2, stride=2)]
    sequence += [nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)]
    sequence += [nn.AvgPool3d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4):
        super(ResNet, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        max_ndf = 4
        conv_layers = [
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool3d(8)]
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output

class Dense_Generator(BasicMoDule):
    def __init__(self):
        BasicMoDule.__init__(self)
        self.init_dim = 64
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.encoder_1_1c = nn.Conv3d(in_channels=3,out_channels=self.init_dim,kernel_size=3,stride=1,padding=1)
        self.encoder_1_2c = nn.Conv3d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=3,stride=1,padding=1)
        self.encoder_1_bn = nn.InstanceNorm3d(self.init_dim) #
        self.encoder_1_relu = nn.LeakyReLU(0.2)

        self.encoder_2_1c = nn.Conv3d(in_channels=self.init_dim, out_channels=self.init_dim*2, kernel_size=3, stride=1, padding=1)
        self.encoder_2_2c = nn.Conv3d(in_channels=self.init_dim*3, out_channels=self.init_dim * 2, kernel_size=3, stride=1, padding=1)
        self.encoder_2_bn = nn.InstanceNorm3d(self.init_dim * 2)
        self.encoder_2_relu = nn.LeakyReLU(0.2)

        self.encoder_3_1c = nn.Conv3d(in_channels=self.init_dim*5, out_channels=self.init_dim*4, kernel_size=3, stride=1, padding=1)
        self.encoder_3_2c = nn.Conv3d(in_channels=self.init_dim * 9, out_channels=self.init_dim * 4, kernel_size=3, stride=1, padding=1)
        self.encoder_3_bn = nn.InstanceNorm3d(self.init_dim*4)
        self.encoder_3_relu = nn.LeakyReLU(0.2)

        self.encoder_4_1c = nn.Conv3d(in_channels=self.init_dim *13, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_4_2c = nn.Conv3d(in_channels=self.init_dim * 21, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_4_bn = nn.InstanceNorm3d(self.init_dim*8)
        self.encoder_4_relu = nn.LeakyReLU(0.2)

        self.encoder_5_1c = nn.Conv3d(in_channels=self.init_dim*29, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_5_2c = nn.Conv3d(in_channels=self.init_dim * 37, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_5_bn = nn.InstanceNorm3d(self.init_dim*8)
        self.encoder_5_relu = nn.LeakyReLU(0.2)

        self.encoder_6_1c = nn.Conv3d(in_channels=self.init_dim*45, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_6_2c = nn.Conv3d(in_channels=self.init_dim * 53, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_6_bn = nn.InstanceNorm3d(self.init_dim*8)
        self.encoder_6_relu = nn.LeakyReLU(0.2)

        self.encoder_7_1c = nn.Conv3d(in_channels=self.init_dim*61, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_7_2c = nn.Conv3d(in_channels=self.init_dim * 69, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_7_bn = nn.InstanceNorm3d(self.init_dim*8)
        self.encoder_7_relu = nn.LeakyReLU(0.2)

        self.encoder_8_1c = nn.Conv3d(in_channels=self.init_dim * 77, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_8_2c = nn.Conv3d(in_channels=self.init_dim * 85, out_channels=self.init_dim * 8, kernel_size=3, stride=1,padding=1)
        self.encoder_8_bn = nn.InstanceNorm3d(self.init_dim*8)
        self.encoder_8_relu = nn.ReLU()

        self.decoder_1_dc = nn.ConvTranspose3d(in_channels=self.init_dim*93, out_channels=self.init_dim*8,kernel_size=2,padding=0,stride=2)
        self.decoder_1_1c = nn.Conv3d(in_channels=self.init_dim * 85, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_1_2c = nn.Conv3d(in_channels=self.init_dim * 93, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_1_bn = nn.InstanceNorm3d(self.init_dim * 8)
        self.decoder_1_relu = nn.ReLU()

        self.decoder_2_dc = nn.ConvTranspose3d(in_channels=self.init_dim*101,out_channels=self.init_dim*8,kernel_size=2,padding=0,stride=2)
        self.decoder_2_1c = nn.Conv3d(in_channels=self.init_dim * 69, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_2_2c = nn.Conv3d(in_channels=self.init_dim * 77, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_2_bn = nn.InstanceNorm3d(self.init_dim * 8)
        self.decoder_2_relu = nn.ReLU()

        self.decoder_3_dc = nn.ConvTranspose3d(in_channels=self.init_dim*85,out_channels=self.init_dim*8,kernel_size=2,padding=0,stride=2)
        self.decoder_3_1c = nn.Conv3d(in_channels=self.init_dim * 53, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_3_2c = nn.Conv3d(in_channels=self.init_dim * 61, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_3_bn = nn.InstanceNorm3d(self.init_dim * 8)
        self.decoder_3_relu = nn.ReLU()

        self.decoder_4_dc = nn.ConvTranspose3d(in_channels=self.init_dim*69,out_channels=self.init_dim*8,kernel_size=2,padding=0,stride=2)
        self.decoder_4_1c = nn.Conv3d(in_channels=self.init_dim * 37, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_4_2c = nn.Conv3d(in_channels=self.init_dim * 45, out_channels=self.init_dim * 8, kernel_size=3, stride=1, padding=1)
        self.decoder_4_bn = nn.InstanceNorm3d(self.init_dim * 8)
        self.decoder_4_relu = nn.ReLU()

        self.decoder_5_dc = nn.ConvTranspose3d(in_channels=self.init_dim*53, out_channels=self.init_dim*4,kernel_size=2,padding=0,stride=2)
        self.decoder_5_1c = nn.Conv3d(in_channels=self.init_dim * 17, out_channels=self.init_dim * 4, kernel_size=3, stride=1, padding=1)
        self.decoder_5_2c = nn.Conv3d(in_channels=self.init_dim * 21, out_channels=self.init_dim * 4, kernel_size=3, stride=1, padding=1)
        self.decoder_5_bn = nn.InstanceNorm3d(self.init_dim * 4)
        self.decoder_5_relu = nn.ReLU()

        self.decoder_6_dc = nn.ConvTranspose3d(in_channels=self.init_dim *25, out_channels=self.init_dim*2,kernel_size=2,padding=0,stride=2)
        self.decoder_6_1c = nn.Conv3d(in_channels=self.init_dim * 8, out_channels=self.init_dim * 2, kernel_size=3, stride=1, padding=1)
        self.decoder_6_2c = nn.Conv3d(in_channels=self.init_dim * 10, out_channels=self.init_dim * 2, kernel_size=3, stride=1, padding=1)
        self.decoder_6_bn = nn.InstanceNorm3d(self.init_dim * 2)
        self.decoder_6_relu = nn.ReLU()

        self.decoder_7_dc = nn.ConvTranspose3d(in_channels=self.init_dim *12, out_channels=self.init_dim,kernel_size=2,padding=0,stride=2)
        self.decoder_7_1c = nn.Conv3d(in_channels=self.init_dim * 2, out_channels=self.init_dim, kernel_size=3, stride=1, padding=1)
        self.decoder_7_2c = nn.Conv3d(in_channels=self.init_dim*3, out_channels=self.init_dim, kernel_size=3, stride=1, padding=1)
        self.decoder_7_bn = nn.InstanceNorm3d(self.init_dim)
        self.decoder_7_relu = nn.ReLU()

        self.decoder_8_dc = nn.ConvTranspose3d(in_channels=self.init_dim*4, out_channels=3, kernel_size=2,padding=0,stride=2)
        self.decoder_8_tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0.,std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.encoder_1_1c(x)
        e1 = self.encoder_1_relu(e1)
        e1 = self.encoder_1_2c(e1)
        e1 = self.maxpool(e1)

        e2_i1 = self.encoder_1_relu(e1)
        e2_c1 = self.encoder_2_1c(e2_i1)
        e2_c1 = self.encoder_2_bn(e2_c1)
        e2_c1 = self.encoder_2_relu(e2_c1)
        e2_i2 = t.cat((e2_i1,e2_c1), dim=1)
        e2_c2 = self.encoder_2_2c(e2_i2)
        e2_c2 = self.encoder_2_bn(e2_c2)
        e2 = t.cat((e2_i2, e2_c2), dim=1)
        e2 = self.maxpool(e2)

        e3_i1 = self.encoder_2_relu(e2)
        e3_c1 = self.encoder_3_1c(e3_i1)
        e3_c1 = self.encoder_3_bn(e3_c1)
        e3_c1 = self.encoder_3_relu(e3_c1)
        e3_i2 = t.cat((e3_i1, e3_c1), dim=1)
        e3_c2 = self.encoder_3_2c(e3_i2)
        e3_c2 = self.encoder_3_bn(e3_c2)
        e3= t.cat((e3_i2, e3_c2), dim=1)
        e3 = self.maxpool(e3)

        e4_i1 = self.encoder_3_relu(e3)
        e4_c1 = self.encoder_4_1c(e4_i1)
        e4_c1 = self.encoder_4_bn(e4_c1)
        e4_c1 = self.encoder_4_relu(e4_c1)
        e4_i2 = t.cat((e4_i1, e4_c1), dim=1)
        e4_c2 = self.encoder_4_2c(e4_i2)
        e4_c2 = self.encoder_4_bn(e4_c2)
        e4 = t.cat((e4_i2, e4_c2), dim=1)
        e4 = self.maxpool(e4)

        e5_i1 = self.encoder_4_relu(e4)
        e5_c1 = self.encoder_5_1c(e5_i1)
        e5_c1 = self.encoder_5_bn(e5_c1)
        e5_c1 = self.encoder_5_relu(e5_c1)
        e5_i2 = t.cat((e5_i1, e5_c1), dim=1)
        e5_c2 = self.encoder_5_2c(e5_i2)
        e5_c2 = self.encoder_5_bn(e5_c2)
        e5 = t.cat((e5_i2, e5_c2), dim=1)
        e5 = self.maxpool(e5)

        e6_i1 = self.encoder_5_relu(e5)
        e6_c1 = self.encoder_6_1c(e6_i1)
        e6_c1 = self.encoder_6_bn(e6_c1)
        e6_c1 = self.encoder_6_relu(e6_c1)
        e6_i2 = t.cat((e6_i1, e6_c1), dim=1)
        e6_c2 = self.encoder_6_2c(e6_i2)
        e6_c2 = self.encoder_6_bn(e6_c2)
        e6 = t.cat((e6_i2, e6_c2), dim=1)
        e6 = self.maxpool(e6)

        e7_i1 = self.encoder_6_relu(e6)
        e7_c1 = self.encoder_7_1c(e7_i1)
        e7_c1 = self.encoder_7_bn(e7_c1)
        e7_c1 = self.encoder_7_relu(e7_c1)
        e7_i2 = t.cat((e7_i1, e7_c1), dim=1)
        e7_c2 = self.encoder_7_2c(e7_i2)
        e7_c2 = self.encoder_7_bn(e7_c2)
        e7 = t.cat((e7_i2, e7_c2), dim=1)
        e7 = self.maxpool(e7)

        e8_i1 = self.encoder_7_relu(e7)
        e8_c1 = self.encoder_8_1c(e8_i1)
        e8_c1 = self.encoder_8_bn(e8_c1)
        e8_c1 = self.encoder_8_relu(e8_c1)
        e8_i2 = t.cat((e8_i1, e8_c1), dim=1)
        e8_c2 = self.encoder_8_2c(e8_i2)
        e8_c2 = self.encoder_8_bn(e8_c2)
        e8 = t.cat((e8_i2, e8_c2), dim=1)
        e8 = self.maxpool(e8)


        d1 = self.encoder_8_relu(e8)
        d1 = self.decoder_1_dc(d1)
        d1 = F.dropout(d1, 0.5, training=True)
        d1_i1 = t.cat((e7, d1), dim=1)
        d1_c1 = self.decoder_1_1c(d1_i1)
        d1_c1 = self.decoder_1_bn(d1_c1)
        d1_c1 = self.decoder_1_relu(d1_c1)
        d1_i2 = t.cat((d1_i1, d1_c1),dim=1)
        d1_c2 = self.decoder_1_2c(d1_i2)
        d1_c2 = self.decoder_1_bn(d1_c2)
        d1 = t.cat((d1_i2, d1_c2), dim=1)

        d2 = self.decoder_1_relu(d1)
        d2 = self.decoder_2_dc(d2)
        d2 = F.dropout(d2, 0.5, training=True)
        d2_i1 = t.cat((e6, d2), dim=1)
        d2_c1 = self.decoder_2_1c(d2_i1)
        d2_c1 = self.decoder_2_bn(d2_c1)
        d2_c1 = self.decoder_2_relu(d2_c1)
        d2_i2 = t.cat((d2_i1, d2_c1), dim=1)
        d2_c2 = self.decoder_2_2c(d2_i2)
        d2_c2 = self.decoder_2_bn(d2_c2)
        d2 = t.cat((d2_i2, d2_c2), dim=1)

        d3 = self.decoder_2_relu(d2)
        d3 = self.decoder_3_dc(d3)
        d3 = F.dropout(d3, 0.5, training=True)
        d3_i1 = t.cat((e5, d3), dim=1)
        d3_c1 = self.decoder_3_1c(d3_i1)
        d3_c1 = self.decoder_3_bn(d3_c1)
        d3_c1 = self.decoder_3_relu(d3_c1)
        d3_i2 = t.cat((d3_i1, d3_c1), dim=1)
        d3_c2 = self.decoder_3_2c(d3_i2)
        d3_c2 = self.decoder_3_bn(d3_c2)
        d3 = t.cat((d3_i2, d3_c2), dim=1)

        d4 = self.decoder_3_relu(d3)
        d4 = self.decoder_4_dc(d4)
        d4_i1 = t.cat((e4, d4), dim=1)
        d4_c1 = self.decoder_4_1c(d4_i1)
        d4_c1 = self.decoder_4_bn(d4_c1)
        d4_c1 = self.decoder_4_relu(d4_c1)
        d4_i2 = t.cat((d4_i1, d4_c1), dim=1)
        d4_c2 = self.decoder_4_2c(d4_i2)
        d4_c2 = self.decoder_4_bn(d4_c2)
        d4 = t.cat((d4_i2, d4_c2), dim=1)

        d5 = self.decoder_4_relu(d4)
        d5 = self.decoder_5_dc(d5)
        d5_i1 = t.cat((e3, d5), dim=1)
        d5_c1 = self.decoder_5_1c(d5_i1)
        d5_c1 = self.decoder_5_bn(d5_c1)
        d5_c1 = self.decoder_5_relu(d5_c1)
        d5_i2 = t.cat((d5_i1, d5_c1), dim=1)
        d5_c2 = self.decoder_5_2c(d5_i2)
        d5_c2 = self.decoder_5_bn(d5_c2)
        d5 = t.cat((d5_i2, d5_c2), dim=1)

        d6 = self.decoder_5_relu(d5)
        d6 = self.decoder_6_dc(d6)
        d6_i1 = t.cat((e2, d6), dim=1)
        d6_c1 = self.decoder_6_1c(d6_i1)
        d6_c1 = self.decoder_6_bn(d6_c1)
        d6_c1 = self.decoder_6_relu(d6_c1)
        d6_i2 = t.cat((d6_i1, d6_c1), dim=1)
        d6_c2 = self.decoder_6_2c(d6_i2)
        d6_c2 = self.decoder_6_bn(d6_c2)
        d6 = t.cat((d6_i2, d6_c2), dim=1)

        d7 = self.decoder_6_relu(d6)
        d7 = self.decoder_7_dc(d7)
        d7_i1 = t.cat((e1, d7), dim=1)
        d7_c1 = self.decoder_7_1c(d7_i1)
        d7_c1 = self.decoder_7_bn(d7_c1)
        d7_c1 = self.decoder_7_relu(d7_c1)
        d7_i2 = t.cat((d7_i1, d7_c1), dim=1)
        d7_c2 = self.decoder_7_2c(d7_i2)
        d7_c2 = self.decoder_7_bn(d7_c2)
        d7 = t.cat((d7_i2, d7_c2), dim=1)

        d8 = self.decoder_7_relu(d7)
        d8 = self.decoder_8_dc(d8)
        d8 = self.decoder_8_tanh(d8)

        return d8


class Patch_Discriminator(BasicMoDule):
    def __init__(self):
        BasicMoDule.__init__(self)
        self.init_dim  = 32
        self.discriminator = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=self.init_dim, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=self.init_dim, out_channels=self.init_dim*2, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(self.init_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=self.init_dim*2, out_channels=self.init_dim*4, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(self.init_dim*4),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=self.init_dim * 8, out_channels=1, kernel_size=3, padding=1, stride=1), # 32-32
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0.,std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.discriminator(x)
        return score













