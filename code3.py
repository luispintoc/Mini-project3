import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

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

train_images = np.asarray(normalization(train_images))
test_images = np.asarray(normalization(test_images))

features_numpy = train_images
targets_numpy = train_labels
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 44) 
learning_rate = 0.0005
train_batch_size = 1600
test_batch_size = 1600

X_train = torch.from_numpy(features_train)
X_test = torch.from_numpy(features_test)

Y_train = torch.from_numpy(targets_train).type(torch.LongTensor) 
Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)


train = torch.utils.data.TensorDataset(X_train,Y_train)
test = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = train_batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = test_batch_size, shuffle = False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=0)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(5408, 64) 
        # self.fc2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10) 

    def forward(self,x):
        
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        # out = self.fc2(out)
        # out = self.dropout(out)
        out = self.out(out)
        
        return out


model = CNN()
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

epochs = 250
train_losses, test_losses = [] ,[]
t_start = time.time()

ep_count = 0
acc_tolerance = 0.005
acc_norm = 0

for epoch in range(epochs):
    start = time.time()
    running_loss = 0
    for images,labels in train_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        train = Variable(images.view(-1,1,64,64))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        output = model(train)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    else:
        test_loss = 0
        acc_old = acc_norm
        accuracy = 0

        with torch.no_grad(): #Turning off gradients to speed up
            model.eval()
            for images,labels in test_loader:
                if use_cuda:
                    images, labels = images.cuda(), labels.cuda()
                test = Variable(images.view(-1,1,64,64))
                labels = Variable(labels)
                
                log_ps = model(test)
                test_loss += criterion(log_ps,labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        model.train()        
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        end = time.time()
        epoch_runtime = end - start
        
        acc_norm = accuracy/len(test_loader)
        
        d_acc_norm = abs(acc_norm - acc_old)
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
      		"Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
      		"Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      		"Test Accuracy: {:.3f}".format(accuracy/len(test_loader)),
              "Epoch Runtime: {:.3f}".format(epoch_runtime))
        
        if d_acc_norm < acc_tolerance:
            ep_count += 1
        else:
            ep_count = 0
            
        if ep_count > 10:
            print("Ending Training at Epoch {}".format(epoch))
            break

t_end = time.time()

print("Total Runtime: {:.3f}".format(t_end - t_start))

plt.figure(1)
plt.plot(train_losses, label='Training loss')
plt.figure(2)
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
