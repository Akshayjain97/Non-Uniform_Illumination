from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import argparse
import sys
from models import *
from utils import progress_bar
import numpy as np
from PIL import Image
from numpy.random import choice
from masking_functions import *
import pandas as pd


parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training')
#parser.add_argument('--lr', default=0.0001, type=float, help='learning rate'); lr1 = '0001'
parser.add_argument('--lr', default=0.001, type=float, help='learning rate'); lr1 = '001'
#parser.add_argument('--lr', default=0.01, type=float, help='learning rate'); lr1 = '01'
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--data_dir', default='/content/Caltech256', type=str, help='path to data')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class Illuminate():
  def __call__(self, img):
    
    chance=choice([0,1],p=[0.2,0.8])
    if(chance):
        mask_func=['one','two','three','four','five','six','seven','eight','nine','ten']
        call_func=choice(mask_func)
        k_values=[-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0,1.2]
        k_selected=choice(k_values)
        mask=eval(call_func+'(k_selected)')
    
        image=np.array(img)

        image=np.add(mask,image)

        image=np.clip(image,0,255)
   
        image=np.uint8(image)

        image=Image.fromarray(image,'RGB')    
        return image
    return img




# Data
print('==> Preparing data..')
data_dir = args.data_dir
transform_train = transforms.Compose([
    # Illuminate(),
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#bs = 32 #set batch size
bs = 64 #set batch size
#bs = 128 #set batch size


class ImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        #self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx][0]
        
        image = Image.open(img_path).convert("RGB")
        temp = img_path.split('/')[5]
        label = int(temp.split('.')[0])
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label)
        return image, label




training_data= ImageDataset('/content/Train_labels.csv',transform=transform_train)
trainloader = DataLoader(training_data, batch_size=bs, shuffle=True,num_workers=2)


test_data= ImageDataset('/content/Test_labels.csv',transform=transform_test)
testloader = DataLoader(test_data, batch_size=bs, shuffle=False,num_workers=2)


# Model
print('==> Building model..')
net = VGG('VGG16'); net1 = 'vgg16'
# net = ResNet18(); net1 = 'ResNet18' 

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'

if args.resume:
    path_to_file = '/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'_intermediate.t7'
    # if os.path.exists(path_to_file):
    # Load checkpoint.
    path_to_best_file = '/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.t7'

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path_to_file)
    best_checkpoint = torch.load(path_to_best_file)
    net.load_state_dict(best_checkpoint['net'])
    best_acc = best_checkpoint['acc']
    start_epoch = checkpoint['epoch']

if not os.path.isdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults'):
  os.mkdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults')

# Training
def train(epoch):
    f = open('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    f.close()

def test(epoch):
    f = open('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
    global best_acc
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
            if (batch_idx + 1) == len(testloader):
                f.write('Test | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                    % (epoch, test_loss / (batch_idx + 1), 100. * correct / total))
    f.close()
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults'):
        os.mkdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults')
    torch.save(state, '/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'_intermediate.t7')
        
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults'):
            os.mkdir('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults')
        torch.save(state, '/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.t7')
        best_acc = acc

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1, last_epoch=-1)

for epoch in range(start_epoch, 150):
    scheduler.step()
    train(epoch)
    test(epoch)

f = open('/content/gdrive/MyDrive/NUI/Caltech256/CheckpointsResults/Caltech256_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
f.write('Best Accuracy:  %.3f\n'
    % (best_acc))
f.close()

print("Best Accuracy: ", best_acc)
