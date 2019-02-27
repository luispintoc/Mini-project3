import torch, torchvision
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mlxtend.data import loadlocal_mnist
from matplotlib.patches import Rectangle
import os

x, y = loadlocal_mnist(images_path = 'mnist_data/raw/train-images-idx3-ubyte',	#x.shape=(60000,784) -> 60000 images and 784 pixels per image (28x28)
						labels_path = 'mnist_data/raw/train-labels-idx1-ubyte')
w,h = 28,28

print(x.shape[0])
x = [x[i].reshape(w,h) for i in range(x.shape[0])]

# plt.imshow(x[1], cmap = 'gray')
# plt.show()
# cv2.imshow('s',x[1])
# k = cv2.waitKey(0)

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

print(bbox2(x[1]))