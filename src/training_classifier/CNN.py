import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import MyDataset 
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
 
        self.conv1 = torch.nn.Conv2d( 2, 18, kernel_size=3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(18, 36, kernel_size=3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(36, 50, kernel_size=3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(50, 50, kernel_size=3, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(2700, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = x.view(-1, np.prod(x.size()[1:]))
        last = torch.sigmoid(self.fc1(x))
        x = self.fc2(last).squeeze()
        return x, last

def save_checkpoint(save_counter, model, optimizer, prefix):
    state = {'save_counter': save_counter, 'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state, '%s-%d' % (prefix, save_counter))

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    save_counter = checkpoint['save_counter']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (save_counter {})".format(filename, save_counter))
    return model, optimizer, save_counter

