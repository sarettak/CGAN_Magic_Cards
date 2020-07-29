''' Improving the Improved Training of Wasserstein GANs: A Consistency Term and Its Dual Effect '''
import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from models import Discriminator, Generator
from dataset import MagicDataset, MagicTransform

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--gamma', type=float, default=0.0003, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: beta 1')
parser.add_argument('--b2', type=float, default=0.999, help='adam: beta 2')
parser.add_argument('--n_critic', type=int, default=5, help='number of critic iterations per generator iteration')
parser.add_argument('--lambda_1', type=int, default=10, help='gradient penalty coefficient')
parser.add_argument('--lambda_2', type=int, default=2, help='consistency term coefficient')
parser.add_argument('--M', type=float, default=0.2, help='ct-gan hyperparameter')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=128, help='number of input units to generate an image')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=256, help='interval betwen image samples')
parser.add_argument('--load_in_mem', action='store_true', default=False, help='load training images in memory')
opt = parser.parse_args()

try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() # Create tensorboard env.
except ImportError:
    writer = None
else:
    writer.close() # Clear all figures.

train_transform = [transforms.Resize([opt.img_size, opt.img_size])]
norm_mean = [0.5,0.5,0.5]
norm_std = [0.5,0.5,0.5]
# 0 No Augmentation
train_transform = [transforms.Resize([opt.img_size, opt.img_size])]

# # 1 Augmentation Techniques (Color)
# train_transform = [MagicTransform([transforms.ColorJitter(contrast=2)]), 
#                     transforms.Resize([image_size, image_size])]

# # 2 Augmentation Techniques (Color + Position)
# train_transform = [MagicTransform([transforms.RandomHorizontalFlip(p=1),
#                     transforms.ColorJitter(saturation=2)]),
#                     transforms.Resize([image_size, image_size])]

train_transform = transforms.Compose(train_transform + [
                     transforms.ToTensor(),
                     transforms.Normalize(norm_mean, norm_std)])

magic = MagicDataset("data/magic", transform=train_transform, load_in_mem=opt.load_in_mem)

batch_iterator = DataLoader(magic, shuffle=True, batch_size=opt.batch_size) # List, NCHW format.

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
gan_loss = nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.gamma, betas=(opt.b1, opt.b2))
optimizer_G = optim.Adam(generator.parameters(), lr=opt.gamma, betas=(opt.b1, opt.b2))

# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

noise_fixed = Variable(Tensor(25, opt.latent_dim, 1, 1).normal_(0, 1), requires_grad=False) # To track the progress of the GAN.

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):
        # == Discriminator update == #
        for iter in range(opt.n_critic):
            # Sample real and fake images, using notation in paper.
            x = Variable(batch.type(Tensor))
            noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1))
            x_tilde = Variable(generator(noise), requires_grad=True)

            epsilon = Variable(Tensor(batch.size(0), 1, 1, 1).uniform_(0, 1))

            x_hat = epsilon*x + (1 - epsilon)*x_tilde
            x_hat = torch.autograd.Variable(x_hat, requires_grad=True)

            # Put the interpolated data through critic.
            dw_x = discriminator(x_hat)
            grad_x = torch.autograd.grad(outputs=dw_x, inputs=x_hat,
                                         grad_outputs=Variable(Tensor(batch.size(0), 1, 1, 1).fill_(1.0), requires_grad=False),
                                         create_graph=True, retain_graph=True, only_inputs=True)
            grad_x = grad_x[0].view(batch.size(0), -1)
            grad_x = grad_x.norm(p=2, dim=1)

            # Update critic.
            optimizer_D.zero_grad()

            # Standard WGAN loss.
            d_wgan_loss = torch.mean(discriminator(x_tilde)) - torch.mean(discriminator(x))

            # WGAN-GP loss.
            d_wgp_loss = torch.mean((grad_x - 1)**2)

            ###### Consistency term. ######
            dw_x1, dw_x1_i = discriminator(x, dropout=0.5, intermediate_output=True) # Perturb the input by applying dropout to hidden layers.
            dw_x2, dw_x2_i = discriminator(x, dropout=0.5, intermediate_output=True)
            # Using l2 norm as the distance metric d, referring to the official code (paper ambiguous on d).
            second_to_last_reg = ((dw_x1_i-dw_x2_i) ** 2).mean(dim=1).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
            d_wct_loss = (dw_x1-dw_x2) ** 2 \
                         + 0.1 * second_to_last_reg \
                         - opt.M
            d_wct_loss, _ = torch.max(d_wct_loss, 0) # torch.max returns max, and the index of max

            # Combined loss.
            d_loss = d_wgan_loss + opt.lambda_1*d_wgp_loss + opt.lambda_2*d_wct_loss

            d_loss.backward()
            optimizer_D.step()

        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim, 1, 1).normal_(0, 1))
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()

        g_loss = -torch.mean(discriminator(imgs_fake))

        g_loss.backward()
        optimizer_G.step()

        if writer:
            batches_done = epoch * len(batch_iterator) + i
            if batches_done % opt.sample_interval == 0:

                # Generate images for a given set of fixed noise
                # so we can track how the GAN learns.
                imgs_fake_fixed = generator(noise_fixed).detach().data
                imgs_fake_fixed = imgs_fake_fixed.add_(1).div_(2)

                writer.add_scalars("losses", {"dloss": d_loss.item(), "gloss": g_loss.item()}, 
                global_step=epoch)
                # Display results on tensorboard.
                writer.add_images("samples", imgs_fake.data[:24], global_step=epoch)

               