import torch, torchvision
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize((0.1307,),(0.3081,))])

data_train = torchvision.datasets.MNIST('/files/', train=True, download=False,transform=T)
data_test = torchvision.datasets.MNIST('/files/', train=False, download=False,transform=T)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=True)



examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape) #128 images = # batch, single color channel (greyscale) and 28x28 "pixels"
# print(example_targets.shape) #128 targets

#		**** Graph Y images ****
Y = 6
fig = plt.figure()
for i in range(Y):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Target: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

# plt.show()
# img = example_data[1][0]
# sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
# new = cv2.filter2D(img, -1, sobel_y)
# plt.imshow(new, cmap='gray')
# plt.show()


#		**** Neural Net ****
# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net,self).__init__()
# 		self.conv1 = nn 
