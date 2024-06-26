import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, model_type, latent_dim):
        super(Generator, self).__init__()
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.image_shape = {
            'mnist': (1, 28, 28),
            'f_mnist': (1, 28, 28),
            'cifar10': (3, 32, 32),
            'cifar100': (3, 32, 32),
            'celeba': (3, 64, 64),
            'lsun': (3, 64, 64),
            'afhq': (3, 128, 128)
        }
        ngf = 64
        nc = self.image_shape[self.model_type][0]
        nz = latent_dim
        nsize = self.image_shape[self.model_type][1]

        if nsize == 32:
            modules = [
                # (bs, nz, 1, 1) -> (bs, ngf*4, 4, 4)
                View(-1, nz, 1, 1),
                nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
            ]
        elif nsize == 64:
            modules = [
                # (bs, nz, 1, 1) -> (bs, ngf*8, 4, 4)
                View(-1, nz, 1, 1),
                nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(inplace=True),
                # (bs, ngf*8, 4, 4) -> (bs, ngf*4, 8, 8)
                nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            ]
        elif nsize == 128:
            modules = [
                # (bs, nz, 1, 1) -> (bs, ngf*16, 4, 4)
                View(-1, nz, 1, 1),
                nn.ConvTranspose2d(nz, ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(inplace=True),
                # (bs, ngf*16, 4, 4) -> (bs, ngf*8, 8, 8)
                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(inplace=True),
                # (bs, ngf*8, 8, 8) -> (bs, ngf*4, 16, 16)
                nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            ]
        else:
            raise AssertionError()

        modules += [
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            # (bs, ngf*4, 4(8), 4(8)) -> (bs, ngf*2, 8(16), 8(16))
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            # (bs, ngf*2, 8(16), 8(16)) -> (bs, ngf, 16(32), 16(32))
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # (bs, ngf, 16(32), 16(32)) -> (bs, nc, 32(64), 32(64))
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, model_type):
        super(Discriminator, self).__init__()
        self.model_type = model_type
        self.image_shape = {
            'mnist': (1, 28, 28),
            'f_mnist': (1, 28, 28),
            'cifar10': (3, 32, 32),
            'cifar100': (3, 32, 32),
            'celeba': (3, 64, 64),
            'lsun': (3, 64, 64),
            'afhq': (3, 128, 128)
        }
        ndf = 64
        nc = self.image_shape[self.model_type][0]
        ngs = 512
        nsize = self.image_shape[self.model_type][1]

        # DCGAN Discriminator
        modules = [
            # (bs, nc, 32(64), 32(64)) -> (bs, ndf, 16(32), 16(32))
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # (bs, ndf, 16(32), 16(32)) -> (bs, ndf*2, 8(16), 8(16))
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (bs, ndf*2, 8(16), 8(16)) -> (bs, ndf*4, 4(8), 4(8))
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if nsize == 32:
            modules += [
                # Geometric Block
                nn.AvgPool2d(kernel_size=2),
                View(-1, ndf*4),
                nn.Linear(ndf*4, ngs),
            ]
        elif nsize == 64:
            modules += [
                # (bs, ndf*4, 8, 8) -> (bs, ndf*8, 4, 4)
                nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                # Geometric Block
                nn.AvgPool2d(kernel_size=2),
                View(-1, ndf*8),
                nn.Linear(ndf*8, ngs),
            ]
        elif nsize == 128:
            modules += [
                # (bs, ndf*4, 16, 16) -> (bs, ndf*8, 8, 8)
                nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, inplace=True),
                # (bs, ndf*8, 8, 8) -> (bs, ndf*16, 4, 4)
                nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # Geometric Block
                nn.AvgPool2d(kernel_size=2),
                View(-1, ndf*16),
                nn.Linear(ndf*16, ngs),
            ]
        else:
            raise AssertionError()

        self.main = nn.Sequential(*modules)

    def forward(self, input):
        return self.main(input)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)