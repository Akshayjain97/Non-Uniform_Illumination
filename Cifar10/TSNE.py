import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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


# Function to extract features
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            lbls = lbls.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels
    

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
    

    


# Set device

if __name__== '__main__':


    net = models.mobilenet_v3_small(); net1 = 'mobilenet_v3_small'
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 10)
    state = torch.load("/sda/akshay/NUI/Results/CIFAR10_B256_LR001_mobilenet_v3_small_Adam.t7")
    net.load_state_dict(state_dict=state['net'])
    net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])
    net = net.to(device)



    net2 = models.inception_v3();#net2 = 'inception_v3'
    net2.fc = nn.Linear(net2.fc.in_features, 10)

    # # Modify the auxiliary classifier if it exists
    if net2.aux_logits:
        net2.AuxLogits.fc = nn.Linear(net2.AuxLogits.fc.in_features, 10)
    state2 = torch.load("/sda/akshay/NUI/Results/CIFAR10_B128_LR001_inception_v3_Adam.t7")
    net2.load_state_dict(state_dict=state2['net'])

    net2.fc = nn.Sequential(*list(net2.fc.children())[:-1])
    net2 = net2.to(device)

    for m in range(1):#,13):
        for k in [-1.2]:#,1.2]:                
            # Load CIFAR-10 dataset
            # mask = masking(m,k)

            transform = transforms.Compose([
                # Illuminate(mask=mask),
                transforms.Resize((224, 224)),  # Models expect 224x224 images
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform2 = transforms.Compose([
                # Illuminate(mask=mask),
                transforms.Resize((299, 299)),  # Models expect 224x224 images
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


            test_dataset2 = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform2)
            test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=128, shuffle=False)


            # Extract features
            features_mobilenet_v3, labels = extract_features(net, test_loader)
            features_inception_v3, _ = extract_features(net2, test_loader2)



            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=50, random_state=42)

            features_mobilenet_v3_2d = tsne.fit_transform(features_mobilenet_v3)
            features_inception_v3_2d = tsne.fit_transform(features_inception_v3)

            # Plot t-SNE
            def plot_tsne(features, labels, title):
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette=sns.color_palette("hsv", 10), legend=None, s=60, alpha=0.7)
                plt.tick_params(axis='both', which='major', labelsize=20)
                # plt.legend(fontsize=20, title_fontsize=14)
                # plt.title(title)
                filename = ''
                if 'MobileNetV3' in title:
                    if k<0:
                        filename = f'mob{m}_neg'
                    else: filename = f'mob{m}_pos'
                else:
                    if k<0:
                        filename = f'inc{m}_neg'
                    else: filename = f'inc{m}_pos'
                plt.savefig(filename)

            plot_tsne(features_mobilenet_v3_2d, labels, f't-SNE of CIFAR-10 using MobileNetV3 Small m={m}, k={k} (Perplexity=50)')
            plot_tsne(features_inception_v3_2d, labels, f't-SNE of CIFAR-10 using InceptionV3 m={m}, k={k} (Perplexity=50)')
