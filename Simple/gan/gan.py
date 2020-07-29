import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from dataset import MagicDataset, MagicTransform

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=512, help='interval betwen image samples')
parser.add_argument('--data_path', type=str, default="../data/magic", help="path of the dataset")
parser.add_argument('--load_in_mem', action='store_true', default=False, help='load training images in memory')

opt = parser.parse_args()


# Set up the visdom, if exists.
# If not, visualization of the training will not be possible.
# If visdom exists, it should be also on.
# Use 'python -m visdom.server' command to activate the server.
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() # Create tensorboard env.
except ImportError:
    writer = None
else:
    writer.close() # Clear all figures.

img_dims = (opt.channels, opt.img_size, opt.img_size)
n_features = opt.channels * opt.img_size * opt.img_size

# This does not directly follow the paper. I did not fully understand the
# order of the architecture they describe in there.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def ganlayer(n_input, n_output, dropout=True):
            pipeline = [nn.Linear(n_input, n_output)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.25))
            return pipeline

        self.model = nn.Sequential(
            *ganlayer(opt.latent_dim, 128, dropout=False),
            *ganlayer(128, 256),
            *ganlayer(256, 512),
            *ganlayer(512, 1024),
            nn.Linear(1024, n_features),
            nn.Tanh()
        )
        # Tanh > Image values are between [-1, 1]

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_dims)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flatimg = img.view(img.size(0), -1)
        prob = self.model( flatimg )
        return prob


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
#                     transfMagicDatasetorms.Resize([image_size, image_size])]

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

optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Loss record.
g_losses = []
d_losses = []
epochs = []
loss_legend = ['Discriminator', 'Generator']

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

for epoch in range(opt.n_epochs):
    print('Epoch {}'.format(epoch))
    for i, (batch, _) in enumerate(batch_iterator):

        real = Variable(Tensor(batch.size(0), 1).fill_(1), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1).fill_(0), requires_grad=False)

        imgs_real = Variable(batch.type(Tensor))
        noise = Variable(Tensor(batch.size(0), opt.latent_dim).normal_(0, 1))
        imgs_fake = Variable(generator(noise), requires_grad=False)


        # == Discriminator update == #
        optimizer_D.zero_grad()

        # A small reminder: given classes c, prob. p, - c*logp - (1-c)*log(1-p) is the BCE/GAN loss.
        # Putting D(x) as p, x=real's class as 1, (..and same for fake with c=0) we arrive to BCE loss.
        # This is intuitively how well the discriminator can distinguish btw real & fake.
        d_loss = gan_loss(discriminator(imgs_real), real) + \
                 gan_loss(discriminator(imgs_fake), fake)

        d_loss.backward()
        optimizer_D.step()


        # == Generator update == #
        noise = Variable(Tensor(batch.size(0), opt.latent_dim).normal_(0, 1))
        imgs_fake = generator(noise)

        optimizer_G.zero_grad()

        # Minimizing (1-log(d(g(noise))) is less stable than maximizing log(d(g)) [*].
        # Since BCE loss is defined as a negative sum, maximizing [*] is == minimizing [*]'s negative.
        # Intuitively, how well does the G fool the D?
        g_loss = gan_loss(discriminator(imgs_fake), real)

        g_loss.backward()
        optimizer_G.step()

        if writer:
            batches_done = epoch * len(batch_iterator) + i
            if batches_done % opt.sample_interval == 0:
                print("Writing on Tensorboard: ")
                # Keep a record of losses for plotting.
                epochs.append(epoch + i/len(batch_iterator))
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                # Display results on tensorboard.

                # vis.line(
                #     X=torch.stack([Tensor(epochs)] * len(loss_legend), dim=1),
                #     Y=torch.stack((Tensor(d_losses), Tensor(g_losses)), dim=1),
                #     opts={
                #         'title': 'loss over time',
                #         'legend': loss_legend,
                #         'xlabel': 'epoch',
                #         'ylabel': 'loss',
                #         'width': 512,
                #         'height': 512
                # },
                #     win=1)
                writer.add_images("samples_1", imgs_fake.data[:24], global_step=batches_done)
              

                # vis.images(
                #     imgs_fake.data[:25],
                #     nrow=5, win=2,
                #     opts={
                #         'title': 'GAN output [Epoch {}]'.format(epoch),
                #         'width': 512,
                #         'height': 512,
                #     }
                # )
