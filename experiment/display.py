from matplotlib import pyplot as plt
import os

folder = "biggan/64"

file_d_loss_fake = 'D_loss_fake.log'
file_d_loss_real = 'D_loss_fake.log'
file_g_loss = 'G_loss.log'

d_loss_fake = []
d_loss_real = []
g_loss = []

for line in open(os.path.join(folder, file_g_loss))


