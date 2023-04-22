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
data_dir = '/content/tiny-imagenet-200'
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

        
        # if(x==0 or x==1):k=-2.2
        # elif(x==2 or x==3):k=-1.8
        # elif(x==4 or x==5):k=-1.4
        # elif(x==6 or x==7):k=-1
        # elif(x==8 or x==9):k=-0.8
        # elif(x==10 or x==11):k=-0.4
        # elif(x==12 or x==13):k=-0.2
        # elif(x==14 or x==15):k=0.0
        # elif(x==16 or x==17):k=0.2
        # elif(x==18 or x==19):k=0.4
        # elif(x==20 or x==21):k=0.6
        # elif(x==22 or x==23):k=0.8
        # elif(x==24 or x==25):k=1.0
        # elif(x==26 or x==27):k=1.4
        # elif(x==28 or x==29):k=1.8
        # elif(x==30 or x==31):k=2.2

        a=k*a
        mask[x][y]=np.array((a,a,a))
        # p,q,r=mask[x][y]
        # if(p>255):p=255
        # if(q>255):q=255
        # if(r>255):r=255
        # if(p<0):p=0
        # if(q<0):q=0
        # if(r<0):r=0 
#         #b=int(b/255)*255+(1-int(b/255))*b
        # mask[x][y]=(p,q,r)
#         mask[x][y]=np.uint8(mask[x][y])



##read image
#img_src = cv2.imread('/content/9654.jpg')

##edge detection filter
# #kernel = np.array([[0.0, -1.0, 0.0], 
#                    [-1.0, 5.0, -1.0],
#                    [0.0, -1.0, 0.0]])

#kernel = kernel/(10*(np.sum(kernel)) if np.sum(kernel)!=0 else 1)

##filter the source image
#mask = cv2.filter2D(img_src,-1,kernel)

##save result image
#cv2.imwrite('/content/res.jpg',mask)



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
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#bs = 32 #set batch size
bs = 64 #set batch size
#bs = 128 #set batch size

        
testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

def load_model(state):
    net.load_state_dict(state['net'])

# Model
print('==> Building model..')
net = VGG('VGG16'); net1 = 'vgg16'
# net = ResNet18(); net1 = 'ResNet18'
#net = ResNet50(); net1 = 'ResNet50'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'
load_model(torch.load("/content/gdrive/MyDrive/AdaNorm-main/Perturbed_TinyImageNet_B64_LR001_vgg16_Adam.t7"))


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
path1='/content/gdrive/MyDrive/AdaNorm-main/result'
path2='Perturbed_Quadrant'
if not os.path.isdir(path1):
    os.mkdir(path1)
path3=os.path.join(path1,path2)
f = open(path3+'.txt', 'a')
f.write('k = %.2f --> Best Accuracy:  %.3f\n'
    % (k,acc))
f.close()


print("Best Accuracy: ", acc)


