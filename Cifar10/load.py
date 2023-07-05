'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import pandas as pd
from torchvision.io import read_image
from models import *
from utils import progress_bar
from diffGrad import diffGrad
from Radam import Radam
from AdaBelief import AdaBelief
from AdamNorm import AdamNorm
from diffGradNorm import diffGradNorm
from RadamNorm import RadamNorm
from AdaBeliefNorm import AdaBeliefNorm
import numpy as np
from PIL import Image
import cv2
import sys

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('k', type=float, help='learning rate');
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate'); lr1 = '001'
# #parser.add_argument('--lr', default=0.01, type=float, help='learning rate'); lr1 = '01'
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# # Data
mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
k=sys.argv[1]
k=float(k)
u,v=32,32

for x in range(0,32):
    for y in range(0,32):
        a=0
        # if(0<=x<=5 or 10<=x<=15 or 20<=x<=25 or 30<=x<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
        # else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))


        # if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
        # if(x<=16 and y>16):a=((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
        # if(x>16 and y<=16):a=-((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
        # if(x>16 and y>16):a=-((x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v)))
        
        # if(0<=y<=5 or 10<=y<=15 or 20<=y<=25 or 30<=y<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
        # else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
        
        if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
        if(x<=16 and y>16):a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
        if(x>16 and y<=16):a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
        if(x>16 and y>16):a=(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))

        # a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
        # a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
        # a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
        # a=(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))
        # a=abs(16-x)*abs(16-y)
        # a=144-abs(16-x)*abs(16-y)
        # a=100-abs(16-x)*abs(16-y)
        # a=50-abs(16-x)*abs(16-y)
        # if(pow(16-x,2)+pow(16-y,2)<=144):a=50-abs(16-x)*abs(16-y)

        
      

        a=k*a
        mask[x][y]=np.array((a,a,a))


class Illuminate():
  def __call__(self, img):

    image=np.array(img)
    # image1=image
    image=np.add(mask,image)
    # image2=image
    # print(np.subtract(image1,image2))
    image=np.clip(image,0,255)
    # image2=image
    # print(np.subtract(image1,image2))
    image=np.uint8(image)
    # image2=image
    # print(np.subtract(image1,image2))
    image=Image.fromarray(image,'RGB')    
    return image
    

print('==> Preparing data..')
transform_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.ConvertImageDtype(torch.float),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
   Illuminate(),
   transforms.ToTensor(),
  #  transforms.ConvertImageDtype(torch.float),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#bs = 32 #set batch size
bs = 64 #set batch size
#bs = 128 #set batch size

        
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

def load_model(state):
    net.load_state_dict(state['net'])

# Model
print('==> Building model..')
# net = VGG('VGG16'); net1 = 'vgg16'
net = ResNet18(); net1 = 'ResNet18'
#net = ResNet50(); net1 = 'ResNet50'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'
load_model(torch.load("/content/gdrive/MyDrive/NUI/Perturbed_CIFAR10_B64_LR001_ResNet18_Adam.t7"))


net.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
         inputs, targets = inputs.to(device), targets.to(device)
         outputs = net(inputs)
         loss = criterion(outputs, targets)
         test_loss += loss.item()
         _, predicted = outputs.max(1)
         total += targets.size(0)
         correct += predicted.eq(targets).sum().item()
    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                

    # Save checkpoint.
    acc = 100. * correct / total
path1='/content/gdrive/MyDrive/NUI/result'
path2='Perturbed_Quadrant'
if not os.path.isdir(path1):
    os.mkdir(path1)
path3=os.path.join(path1,path2)
f = open(path3+'.txt', 'a')
f.write('k = %.2f --> Best Accuracy:  %.3f\n'
    % (k,acc))
f.close()


print("Best Accuracy: ", acc)


