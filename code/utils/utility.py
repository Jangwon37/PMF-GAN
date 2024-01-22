import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting up CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if DEVICE.type == "cuda" else torch.FloatTensor

def createFolder(directory) -> str:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

class ShowImage(nn.Module):
    def __init__(self, train_loader, n_image):
        super(ShowImage, self).__init__()
        self.train_loader = train_loader
        self.n_image = n_image
        real_batch = next(iter(train_loader))
        plt.figure(figsize=(np.sqrt(n_image)+2, np.sqrt(n_image)+2))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:n_image], nrow= int(np.sqrt(n_image)),
                                                 padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

def compute_gradient_penalty(D, real_samples, fake_samples) -> Tensor:
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(DEVICE)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates).view(real_samples.size(0))
    fake = Variable(Tensor(real_samples.shape[0]).to(DEVICE).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 0.5) ** 2).mean()
    return gradient_penalty

def save_samples_from_loader(train_loader, save_dir, num_samples=50000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        if len(os.listdir(save_dir)) > 0:
            print("The directory already contains files. No new samples will be saved.")
            return

    saved_samples = 0
    for i, (images, _) in enumerate(train_loader):
        num_to_save = min(num_samples - saved_samples, images.size(0))
        if num_to_save <= 0:
            break

        for j in range(num_to_save):
            vutils.save_image(images[j], os.path.join(save_dir, f'sample_{saved_samples + j:05d}.png'), normalize=True)

        saved_samples += num_to_save

    print(f'Total {saved_samples} samples saved.')

def save_generated_images(generator, latent_dim, save_dir, num_samples=50000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator.eval()
    with torch.no_grad():
        for i in range(num_samples):
            z = Variable(Tensor(np.random.normal(0, 1, (1, latent_dim))).to(DEVICE))
            gen_img = generator(z)
            file_path = os.path.join(save_dir, f'generated_{i:05d}.png')

            if os.path.exists(file_path):
                os.remove(file_path)

            vutils.save_image(gen_img, file_path, normalize=True)

    print(f'Total {num_samples} generated samples saved.')

def save_discriminator_responses(batches_done, generated_responses, real_responses, save_path):
    filename = os.path.join(save_path, f"{batches_done}.npz")
    np.savez(filename, generated=generated_responses, real=real_responses)