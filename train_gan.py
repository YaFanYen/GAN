import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from model import weights_init, Generator, Discriminator

seed = 999
random.seed(seed)
torch.manual_seed(seed)
root = '/home/ed716/Documents/NewSSD/Cloud Computing/DCGAN-PyTorch/data'
params = {
    "batch_size" : 128,
    'image_size' : 32,
    'nc' : 3,
    'ngf' : 16,
    'ndf' : 16,
    'nz' : 100,
    'epochs' : 20,
    'lrg' : 0.002,
    'lrd' : 0.005,
    'beta1' : 0.5,
    'save_epoch' : 1}

dataset = dset.ImageFolder(root=root,
                           transform=transforms.Compose([
                               transforms.Resize(params['image_size']),
                               transforms.CenterCrop(params['image_size']),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=params['batch_size'], 
                                         shuffle=True, 
                                         num_workers=params['save_epoch'])
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")

sample_batch = next(iter(dataloader))

netG = Generator(params).to(device)
netG.apply(weights_init)
netD = Discriminator(params).to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(params['batch_size'], params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=params['lrd'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lrg'], betas=(params['beta1'], 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(params['epochs']):
    for i, data in enumerate(dataloader, 0):
        # Discriminator
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        netD.zero_grad()
        label = torch.full((batch_size, ), real_label, device=device)
        output = netD(real_data).view(-1)
        label = label.float()
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(batch_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()
        # Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['epochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 100 == 0) or ((epoch == params['epochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1

    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/model_epoch_{}.pth'.format(epoch))

torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/model_final.pth')
'''
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
'''
