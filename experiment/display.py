from matplotlib import pyplot as plt
import os

folder = "experiment_data/biggan/64/"

file_d_loss_fake = 'D_loss_fake.log'
file_d_loss_real = 'D_loss_fake.log'
file_g_loss = 'G_loss.log'

d_loss_fake = []
d_loss_real = []
g_loss = []

with open(os.path.join(folder, file_g_loss)) as f:
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(111)
ax1.set_title("Generator Loss")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.bar(x,y, width=0.7)

fig1=plt.gcf()
plt.show()
plt.draw()

    #print(value)


'''with open('test.txt') as f:
    lines = f.readlines()
    x = [int(line.split()[0]) for line in lines]
    y = [int(line.split()[1]) for line in lines]

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(111)
ax1.set_title("SMP graph")
ax1.set_xlabel('hour')
ax1.set_ylabel('smp')
ax1.bar(x,y, width=0.7)

fig1=plt.gcf()
plt.show()
plt.draw()
'''
