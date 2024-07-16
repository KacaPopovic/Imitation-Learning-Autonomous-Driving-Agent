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
        if self.transform is not None:
            if self.mode == 2:
                if np.random.rand() > 0.8:
                    image = transforms.functional.vflip(image)
                    # Swap labels if vertical flip occurs
                    if label == 1:
                        label = 2
                    elif label == 2:
                        label = 1
                if np.random.rand() > 1:
                    image_np = np.array(image)

                    # Define the grey color range.
                    lower_grey = np.array([100, 100, 100])
                    upper_grey = np.array([200, 200, 200])

                    # Create a mask for grey areas.
                    mask = np.all(image_np >= lower_grey, axis=-1) & np.all(image_np <= upper_grey, axis=-1)

                    # Change grey areas to brown.
                    image_np[mask] = self.brown_color

                    image = Image.fromarray(image_np)

            if self.mode == 5:
                image = transforms.functional.vflip(image)
                if label == 1:
                    label = 2
                elif label == 2:
                    label = 1
            if self.mode == 6:
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
        else:
            self.transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor()])
            if self.mode == 3:
                image = transforms.Grayscale(1)(image)
            image = self.transform(image)
        return image, label


# Updated CNN model with named ReLU layers for fusion
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

        # Add 2nd hidden layer (output layer)
        x = self.fc2(x)
        return x

class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        # Convolutional layer (sees 96x96x3 image tensor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # Convolutional layer (sees 48x48x32 tensor)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.context_length = 32

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64 * 24 * 24, hidden_size=128, batch_first=True)

        # Linear layer (128 -> 5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape the tensor to (batch_size, context_length, 64 * 24 * 24)
        # Reshape the tensor to (batch_size, sequence_length, input_size)
        original_batch_size = x.size(0)
        sequence_length = self.context_length
        batch_size = original_batch_size // sequence_length
        x = x[:batch_size * sequence_length]
        x = x.view(batch_size, sequence_length, 64 * 24 * 24)

        if batch_size == 0:
            x = torch.zeros(original_batch_size, 128)
            return x

        # LSTM forward pass
        x,_ = self.lstm(x)

        # Get the hidden state of the last time step
        #x = h_n[-1]
        x = x.contiguous().view(batch_size * sequence_length, -1)

        # Fully connected layer
        x = self.fc2(x)
        if original_batch_size % sequence_length != 0:
            padding_size = original_batch_size % sequence_length
            padding = torch.zeros(padding_size, x.size(1))
            x = torch.cat((x, padding), dim=0)

        return x