#%% Import and Load Data
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

#%% Run Code

# Define Hyperparameters

learning_rate = 0.0007
train_batch_size = 1000
test_batch_size = 1000
max_epochs = 500
ks = 5 # Kernel Size
pad = 2 # Padding

# Node numbers: lay1 -> lay2 -> lay3 -> 10 digits
lay1 = 1600
lay2 = 400
lay3 = 100

flags_bool = False # If you want warning flags to show up between epochs

# For debugging, set to false for actual training
# Need to rerun load data section after setting from true to false

debug_bool = False

if debug_bool:
    train_images = train_images[0:400]
    train_labels = train_labels[0:400]
    test_images = test_images[0:100]
    train_batch_size = 400
    test_batch_size = 50

# If test_loss > train_loss + test_tolerance a total of test_num consecutive times OR
# The mean of the last 4 test_loss doesn't decrease by test_tolerance a
    # total of test_num consecutive times:
# Terminate
test_tolerance = 0.005 #
test_num = 10

print("Learning Rate: {}, Train Batch: {}, Test Batch: {}".format(learning_rate,
                                                                  train_batch_size,
                                                                  test_batch_size))

print('Test Loss Tolerance: {}, Number of Anomalous Epochs for Termination: {}'.format(test_tolerance,test_num))
print('Kernel Size: {}, Padding: {}, Layers: {} -> {} -> {} -> Classify'.format(ks,
                                                                                pad,
                                                                                lay1,
                                                                                lay2,
                                                                                lay3))

print('------------------------------------------------------------------------')

# Functions

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
        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = ks, stride=1, padding=pad)
        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = ks, stride=1, padding=pad)
        self.cnn_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = ks, stride=1, padding=pad)
        self.cnn_4 = nn.Conv2d(in_channels = 64, out_channels = 100, kernel_size = ks, stride=1, padding=pad)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        
        self.fc1 = nn.Linear(lay1, lay2) 
        self.fc2 = nn.Linear(lay2, lay3) 
        self.out = nn.Linear(lay3, 10) 

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
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

train_losses, test_losses = [] ,[]

acc_norm = 0
loss_count = 0
test_loss_count = 0
test_loss = 0
tlp = 0
tlpp = 0
meantestloss = 0

for epoch in range(max_epochs):
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
        accuracy = 0
        
        tlppp = tlpp
        tlpp = tlp
        tlp = test_loss
        
		test_loss = 0
		
        meantestloss_old = meantestloss
        
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
        
        print("Epoch: {}/{}.. ".format(epoch+1, max_epochs),
      		"Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
      		"Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      		"Test Accuracy: {:.3f}.. ".format(accuracy/len(test_loader)),
              "Epoch Runtime: {:.3f}".format(end - start))
        
        
        # acc_norm = accuracy/len(test_loader)
        # dacc = abs(acc_norm - acc_old)
        dloss = running_loss - test_loss
        
        # Take average of last four epochs
        meantestloss = (test_loss + tlp + tlpp + tlppp)/(4*len(test_loader))
        
        # Change in Mean Test Loss
        dmtl = abs(meantestloss - meantestloss_old)
        
        
        if dloss < 0:
            loss_count += 1
            if flags_bool:
                print("Train < Test Flag")
        else:
            loss_count = 0
        
        if dmtl <  test_tolerance: 
            test_loss_count += 1
            if flags_bool:
                print("No Test Loss Decrease Flag")
        else:
            test_loss_count = 0
        
        if (loss_count > test_num): 
            print("Training ended at Epoch {}/{}".format(epoch+1,max_epochs))
            print("Training Loss < Test Loss for {} Epochs".format(test_num))
            break
        
        if (test_loss_count > test_num):
            print("Training ended at Epoch {}/{}".format(epoch+1,max_epochs))
            print("Test Loss hasn't decreased by {} for {} Epochs".format(test_tolerance,test_num))
            break

plt.figure(1)
plt.plot(train_losses, label='Training loss')
plt.legend(frameon=False)
plt.figure(2)
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()