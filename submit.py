import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#load data
train_images = pd.read_pickle('train_images.pkl')
train_labels = pd.read_csv('train_labels.csv')
test_images = pd.read_pickle('test_images.pkl')

train_labels = np.asarray(train_labels.Category)

def normalization(images):
    pop_mean = []
    pop_std0 = []
    images2 = []

    for image in images:
        batch_mean = (image.mean())
        batch_std0 = (image.std())
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

    pop_mean = (sum(pop_mean)/len(pop_mean))
    pop_std0 = (sum(pop_std0)/len(pop_std0))

    for image in images:
        image = (image - pop_mean)/pop_std0
        images2.append(image)

    return images2

def rescaling(images):
    images2 = []
    for image in images:
        image = rescale(image, 1.5, multichannel = False, mode = 'constant', anti_aliasing = True)
        images2.append(image)
    return images2

train_images = np.asarray(normalization(train_images))
test_images = np.asarray(normalization(test_images))

features_train = train_images
targets_train = train_labels
features_test = test_images

#   **** HYPERPARAMETERS ****

train_batch_size = 1000
test_batch_size = 10000
epochs = 320
lr = 0.0007

#   *************************

X_train = torch.from_numpy(features_train)
X_test = torch.from_numpy(features_test)

Y_train = torch.from_numpy(targets_train).type(torch.LongTensor) 

train = torch.utils.data.TensorDataset(X_train,Y_train)

train_loader = torch.utils.data.DataLoader(train, batch_size = train_batch_size, shuffle = True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 4, stride=1, padding=1)
        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride=1, padding=1)
        self.cnn_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=1, padding=1)
        self.cnn_4 = nn.Conv2d(in_channels = 64, out_channels = 100, kernel_size = 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2d = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(100*9, 300) 
        self.fc2 = nn.Linear(300, 100) 
        self.out = nn.Linear(100, 10) 

    def forward(self,x):
        # print(x.size())
        out = self.cnn_1(x)
        # print(out.size(), 'cn1')
        out = self.relu(out)
        # print(out.size())
        out = self.dropout2d(out)
        # print(out.size())
        out = self.maxpool(out)
        # print(out.size())
        
        out = self.cnn_2(out)
        # print(out.size(), 'cn2')
        out = self.relu(out)
        # print(out.size())
        out = self.dropout2d(out)
        # print(out.size())
        out = self.maxpool(out)
        # print(out.size())

        out = self.cnn_3(out)
        # print(out.size(), 'cn3')
        out = self.relu(out)
        # print(out.size())
        out = self.dropout2d(out)
        # print(out.size())
        out = self.maxpool(out)
        # print(out.size())

        out = self.cnn_4(out)
        # print(out.size())
        out = self.relu(out)
        # print(out.size())
        out = self.dropout2d(out)
        # print(out.size())
        out = self.maxpool(out)
        # print(out.size())
        
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        # print(out.size())
        out = self.dropout(out)
        # print(out.size())
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        
        return out


model = CNN()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

train_losses = []
for epoch in range(epochs):
    running_loss = 0
    for images,labels in train_loader:
        if use_cuda:
            # images = images.float()
            images, labels = images.cuda(), labels.cuda()
        train = Variable(images.view(-1,1,64,64))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        output = model(train)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()


# **** TEST *****

def predict_image(image):
    model.eval()
    input = Variable(image.view(-1,1,64,64))
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

test_labels = []

for images in X_test:
    test_labels.append(predict_image(images))

# print(test_labels)

id = list(range(10000))
ss = list(zip(id,test_labels))

with open('submission_log.csv', 'w', newline = '') as f:
     writer = csv.writer(f, delimiter=',')
     writer.writerows(ss)