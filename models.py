import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 50, kernel_size= 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(50 * 4 * 4, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, input):
        x = F.max_pool2d(F.relu((self.bn1(self.conv1(input)))), 2)
        x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))), 2)
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)

        return x

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        logits = self.fc3(input)

        return logits
