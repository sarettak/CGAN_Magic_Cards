from matplotlib import pyplot as plt
import os

root = "experiment_data/"
file_d_loss_fake = 'D_loss_fake.log'
file_d_loss_real = 'D_loss_real.log'
file_g_loss = 'G_loss.log'

for subdir, dirs, files in os.walk(root):
    for dir_ in dirs:
        if dir_ not in ["bigganhinge", "biggan", "samples", "fixed"]:
            print(subdir)
            with open(os.path.join(subdir, dir_, file_g_loss)) as f:
                lines = f.readlines()
                g_loss = [float(line.split()[1]) for line in lines]

            with open(os.path.join(subdir, dir_, file_d_loss_fake)) as f:
                lines = f.readlines()
                d_loss_fake = [float(line.split()[1]) for line in lines]

            with open(os.path.join(subdir, dir_, file_d_loss_real)) as f:
                lines = f.readlines()
                d_loss_real = [float(line.split()[1]) for line in lines]

            x = range(len(g_loss))

            fig = plt.figure(figsize=(10,10))

            plt.plot(x, g_loss, label="Generator loss")
            plt.plot(x, d_loss_fake, label="Discriminator fake loss")
            plt.plot(x, d_loss_real, label="Discriminator real loss")
            plt.gca().set_xlabel("Batch number")
            plt.gca().set_ylabel("Loss magnitude")
            plt.legend()
            plt.savefig(os.path.join(subdir, dir_, "plot.png"))
