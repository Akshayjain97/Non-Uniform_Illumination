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
from utils import progress_bar

import numpy as np
from PIL import Image
import cv2
import sys
from sklearn.metrics import precision_score, recall_score, f1_score


device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

def masking(m,k):
    u,v=32,32
    print(m,k)
    mask=np.zeros(3072,dtype=np.int32).reshape(32,32,3)
    for x in range(0,32):
        for y in range(0,32):
            a=0
            

            if m==1:a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
            if m==2:a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
            if m==3:a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
            if m==4:a=(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))
            if m==5:a=abs(16-x)*abs(16-y)
            if m==6:a=144-abs(16-x)*abs(16-y)
            if m==7:a=100-abs(16-x)*abs(16-y)
            if m==8:a=50-abs(16-x)*abs(16-y)
            # if(pow(16-x,2)+pow(16-y,2)<=144):a=50-abs(16-x)*abs(16-y)

            if m==9:
                if(0<=y<=5 or 10<=y<=15 or 20<=y<=25 or 30<=y<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
                else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
            
            if m==10:
                if(0<=x<=5 or 10<=x<=15 or 20<=x<=25 or 30<=x<=32):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
                else:a=-((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
            
            if m==11:
                if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
                if(x<=16 and y>16):a=(x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v))
                if(x>16 and y<=16):a=((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
                if(x>16 and y>16):a=(x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v))

            if m==12:
                if(x<=16 and y<=16):a=((u-x)*(30/u))+((v-y)*(30/v))+((u-y)*(20/u))+((v-y)*(20/v))
                if(x<=16 and y>16):a=((x*(30/u))+((v-y)*(30/v))+(y*(20/u))+((v-x)*(20/v)))
                if(x>16 and y<=16):a=-((u-x)*(30/u))+(y*(30/v))+((u-y)*(20/u))+(y*(20/v))
                if(x>16 and y>16):a=-((x*(30/u))+(y*(30/v))+(x*(20/u))+(y*(20/v)))
            

            a=k*a
            mask[x][y]=np.array((a,a,a))
    
    return mask


class Illuminate():
  def __init__(self,mask):
      self.mask = mask
  def __call__(self, img):

    image=np.array(img)
    # image1=image
    mask = self.mask
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
    

if  __name__ == "__main__":
    def load_model(state):
        net.load_state_dict(state['net'])
        # print(state['acc'])

    # Model
    # print('==> Building model..')
    # net = VGG('VGG16'); net1 = 'vgg16'
    # net = ResNet18(); net1 = 'ResNet18'
    #net = ResNet50(); net1 = 'ResNet50'
    # net = torchvision.models.mobilenet_v3_small(); net1 = 'mobilenet_v3_small_perturbed'
    # net.classifier[3] = nn.Linear(net.classifier[3].in_features, 10)

    net = torchvision.models.inception_v3();net1 = 'inception_v3_perturbed'
    net.fc = nn.Linear(net.fc.in_features, 10)

    ## Modify the auxiliary classifier if it exists
    if net.aux_logits:
        net.AuxLogits.fc = nn.Linear(net.AuxLogits.fc.in_features, 10)

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    load_model(torch.load("/sda/akshay/NUI/Results/CIFAR10_B128_LR001_inception_v3_perturbed_Adam.t7"))

    for m in range(1,13):
        for k in [-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2]:
            mask = masking(m,k)

            transform_test = transforms.Compose([
            Illuminate(mask=mask),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            bs = 128 #set batch size
            # bs = 256
                    
            testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

            net.eval()
            test_loss = 0
            correct = 0
            total = 0

            y_true_test = []
            y_pred_test = []

            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_true_test.extend(labels.tolist())
                    y_pred_test.extend(predicted.tolist())
                precision_test = precision_score(y_true_test, y_pred_test, average='micro')
                recall_test = recall_score(y_true_test, y_pred_test, average='micro')
                f1_test = f1_score(y_true_test,y_pred_test,average='micro')
                    

            path1=f'/sda/akshay/NUI/Non-Uniform_Illumination/Cifar10/prf/{net1}'
            path2=f'{m}'
            if not os.path.isdir(path1):
                os.makedirs(path1)
            path3=os.path.join(path1,path2)
            f = open(path3+'_micro.txt', 'a')
            f.write('k = %.2f --> Precison  %.3f | Recall %.3f | F1 %.3f\n'
                % (k,precision_test,recall_test,f1_test))
            f.close()