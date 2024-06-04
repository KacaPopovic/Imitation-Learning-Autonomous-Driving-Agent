from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, mode, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.brown_color = (115, 70, 31)
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.mode == 2:
            if np.random.rand() > 0.8:
                image = transforms.functional.vflip(image)
                # Swap labels if vertical flip occurs
                if label == 1:
                    label = 2
                elif label == 2:
                    label = 1
            if np.random.rand() > 0.8:
                image_np = np.array(image)

                # Define the grey color range.
                lower_grey = np.array([100, 100, 100])
                upper_grey = np.array([200, 200, 200])

                # Create a mask for grey areas.
                mask = np.all(image_np >= lower_grey, axis=-1) & np.all(image_np <= upper_grey, axis=-1)

                # Change grey areas to brown.
                image_np[mask] = self.brown_color

                image = Image.fromarray(image_np)
        image = self.transform(image)
        return image, label


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer (sees 96x96x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # Convolutional layer (sees 48x48x16 tensor)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Linear layer (64 * 12 * 12 -> 500)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)

        # Linear layer (500 -> 5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        # Flatten image input
        x = torch.flatten(x, start_dim=1)

        # Add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

        # Add 2nd hidden layer (output layer)
        x = self.fc2(x)
        return x