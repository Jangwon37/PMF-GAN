import torch.nn as nn

BASE_DIM = 64

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

        def block(in_feat, out_feat, normalize=True):
            layers = [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConvTranspose2d(in_channels=in_feat, out_channels=out_feat, kernel_size=5, stride=1, padding=2, bias=False)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, momentum=0.999, affine=False))
            layers.append(nn.LeakyReLU(inplace=True))
            return layers

        if self.model_type in ['mnist', 'f_mnist']:
            self.first_layer = nn.Linear(in_features=latent_dim, out_features=BASE_DIM * 4 * 7 * 7, bias=False)
            self.model = nn.Sequential(
                *block(BASE_DIM * 4, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM),
                nn.ConvTranspose2d(in_channels=BASE_DIM, out_channels=1, kernel_size=5, stride=1, padding=2,
                                   bias=False),
                nn.Tanh()
            )
        elif self.model_type in ['celeba', 'lsun']:
            self.first_layer = nn.Linear(in_features=latent_dim, out_features=BASE_DIM * 8 * 8 * 8, bias=False)
            self.model = nn.Sequential(
                *block(BASE_DIM * 8, BASE_DIM * 4),
                *block(BASE_DIM * 4, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM),
                nn.ConvTranspose2d(in_channels=BASE_DIM, out_channels=3, kernel_size=5, stride=1, padding=2,
                                   bias=False),
                nn.Tanh()
            )
        elif self.model_type in ['cifar10', 'cifar100']:
            self.first_layer = nn.Linear(in_features=latent_dim, out_features=BASE_DIM * 8 * 4 * 4, bias=False)
            self.model = nn.Sequential(
                *block(BASE_DIM * 8, BASE_DIM * 4),
                *block(BASE_DIM * 4, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM),
                nn.ConvTranspose2d(in_channels=BASE_DIM, out_channels=3, kernel_size=5, stride=1, padding=2,
                                   bias=False),
                nn.Tanh()
            )
        else:
            self.first_layer = nn.Linear(in_features=latent_dim, out_features=BASE_DIM * 8 * 16 * 16, bias=False)
            self.model = nn.Sequential(
                *block(BASE_DIM * 8, BASE_DIM * 4),
                *block(BASE_DIM * 4, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM),
                nn.ConvTranspose2d(in_channels=BASE_DIM, out_channels=3, kernel_size=5, stride=1, padding=2,
                                   bias=False),
                nn.Tanh()
            )


    def forward(self, z):
        reshape_size = {
            'mnist': 7, 'f_mnist': 7,
            'celeba': 8, 'lsun': 8, 'afhq': 16
        }.get(self.model_type, 4)
        img = self.first_layer(z).reshape(z.size(0), -1, reshape_size, reshape_size)
        img = self.model(img)
        img = img.view(img.size(0), *self.image_shape[self.model_type])
        return img


class Discriminator(nn.Module):
    def __init__(self, model_type):
        super(Discriminator, self).__init__()
        self.model_type = model_type

        def block(in_feat, out_feat, stride=2):
            layers = [nn.Conv2d(in_feat, out_feat, 5, stride, 2, bias=True), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        if self.model_type in ['mnist', 'f_mnist']:
            self.model = nn.Sequential(
                *block(1, BASE_DIM),
                *block(BASE_DIM, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM * 4, stride=1),
                nn.AvgPool2d(4)
            )
            self.fc = nn.Linear(in_features=BASE_DIM * 4, out_features=1, bias=False)
        elif self.model_type in ['celeba', 'lsun']:
            self.model = nn.Sequential(
                *block(3, BASE_DIM),
                *block(BASE_DIM, BASE_DIM * 2),
                *block(BASE_DIM * 2, BASE_DIM * 4),
                *block(BASE_DIM * 4, BASE_DIM * 8, stride=1),
                nn.AvgPool2d(8)
            )
            self.fc = nn.Linear(in_features=BASE_DIM * 8, out_features=1, bias=False)
        elif self.model_type in ['cifar10', 'cifar100']:
            self.model = nn.Sequential(
                *block(3, BASE_DIM),
                *block(BASE_DIM, BASE_DIM*2),
                *block(BASE_DIM*2, BASE_DIM*4, stride=1),
                *block(BASE_DIM*4, BASE_DIM*8, stride=1),
                nn.AvgPool2d(8)
            )
            self.fc = nn.Linear(in_features=BASE_DIM * 8, out_features=1, bias=False)
        else:
            self.model = nn.Sequential(
                *block(3, BASE_DIM),
                *block(BASE_DIM, BASE_DIM*2),
                *block(BASE_DIM*2, BASE_DIM*4),
                *block(BASE_DIM*4, BASE_DIM*8),
                nn.AvgPool2d(8)
            )
            self.fc = nn.Linear(in_features=BASE_DIM * 8, out_features=1, bias=False)

    def forward(self, img):
        img = self.model(img).view(img.size()[0], -1)
        validity = self.fc(img)
        return validity

