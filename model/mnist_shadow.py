import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # Image processing part (CNN)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Label processing part (fully connected)
        self.label_fc = nn.Sequential(
            nn.Linear(10, 16),  # Assuming one-hot encoded label
            nn.ReLU()
        )
        
        # Combined processing part
        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7 + 16, 128),  # Adjust the size according to the CNN output
            nn.ReLU(),
            nn.Linear(128, 1)  # Continuous output
        )
        
    def forward(self, x, label):
        # Process the image
        x = self.cnn_layers(x)
        
        # Process the label
        label = self.label_fc(label)
        
        # Concatenate image features and label features
        combined = torch.cat((x, label), dim=1)
        
        # Pass through the fully connected layers
        output = self.fc_layers(combined)
        return output
    
class MNISTLinearRegression(nn.Module):
    def __init__(self):
        super(MNISTLinearRegression, self).__init__()
        
        # MNIST image is 28x28 = 784, one-hot encoded label is 10-dimensional
        self.linear = nn.Linear(784 + 10, 1)
        
    def forward(self, x, one_hot_label):
        # Flatten the MNIST image
        x = x.view(x.size(0), -1)
        
        # Concatenate the image and one-hot encoded label
        combined_input = torch.cat((x, one_hot_label), dim=1)
        
        # Pass the combined input through the linear layer
        output = self.linear(combined_input)
        return output

class ImprovedMNISTModel(nn.Module):
    def __init__(self):
        super(ImprovedMNISTModel, self).__init__()
        
        # Image processing part (CNN)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten()
        )
        
        # Label processing part (fully connected)
        self.label_fc = nn.Sequential(
            nn.Linear(10, 32),  # Assuming one-hot encoded label
            nn.ReLU(),
            nn.Linear(32, 16),  # Adding an extra layer for better feature extraction
            nn.ReLU()
        )
        
        # Combined processing part
        self.fc_layers = nn.Sequential(
            nn.Linear(128*3*3 + 16, 256),  # Adjust the size according to the CNN output
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout to prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Continuous output
        )
        
    def forward(self, x, label):
        # Process the image
        x = self.cnn_layers(x)
        
        # Process the label
        label = self.label_fc(label)
        
        # Concatenate image features and label features
        combined = torch.cat((x, label), dim=1)
        
        # Pass through the fully connected layers
        output = self.fc_layers(combined)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x

class MNISTResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(MNISTResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Fully connected layers for label processing
        self.label_fc = nn.Sequential(
            nn.Linear(num_classes, 64),  # Assuming one-hot encoded label
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fully connected layers for combined processing
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 32, 256),  # Adjust according to the output size from ResNet layers
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Continuous output
        )
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x, label):
        # Process the image through ResNet layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Process the label
        label = self.label_fc(label)
        
        # Concatenate the image features and label features
        combined = torch.cat((x, label), dim=1)
        
        # Fully connected layers to output the continuous variable
        output = self.fc(combined)
        return output


def ResNet18_mnist(num_classes=10):
    return MNISTResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)

