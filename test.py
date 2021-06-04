import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from model import Generator
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth')
parser.add_argument('-num_output', default=50000)
parser.add_argument('-save_path', default='/home/ed716/Documents/NewSSD/Cloud Computing/DCGAN-PyTorch/309511051_img/')
args = parser.parse_args()

state_dict = torch.load(args.load_path)
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

for i in range (50000):
	img = generated_img[i]
	img = np.array(img)
	img = np.transpose(img, (2, 1, 0))
	img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
	name = args.save_path + str(i) + '.png'
	cv.imwrite(name, img)
