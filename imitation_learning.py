import requests
from IPython.display import Image, display
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns


def transform_label(label):
    if label == [0.0, 0.0, 0.0]:
        return 0
    elif label == [-1.0, 0.0, 0.0]:
        return 1
    elif label == [1.0, 0.0, 0.0]:
        return 2
    elif label == [0.0, 1.0, 0.0]:
        return 3
    elif label == [0.0, 0.0, 0.8]:
        return 4
    else:
        return 3  # Default to 0 if none of the above conditions match


# Define a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.brown_color = (115, 70, 31)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
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
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((96, 96))(image)

        return image, label


# Function to visualize the first 30 images with labels
def visualize_images(dataloader, num_images=30):
    plt.figure(figsize=(15, 15))
    images_shown = 0
    for images, labels in dataloader:
        for i in range(images.shape[0]):
            if images_shown >= num_images:
                break
            ax = plt.subplot(6, 5, images_shown + 1)
            image = images[i].permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.title(str(labels[i].item()))
            plt.axis("off")
            images_shown += 1
        if images_shown >= num_images:
            break
    plt.show()


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


# Function for training and validation

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, patience = 3):
    best_val_accuracy = 0.0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    non_improve_counter = 0

    for epoch in range(num_epochs):
        # If the counter reaches the patience limit, stop training
        if non_improve_counter >= patience:
            print("Early stopping due to no improvement")
            break

        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            labels = labels.long()  # Convert labels to long
            images = np.copy(images)  # Make a copy to ensure it is writable
            images = torch.from_numpy(images)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(
                0)  # item extract float from tensor, then we multiply with batch size
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / total_train)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_val = 0
        correct_val = 0
        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:
                labels = labels.long()  # Convert labels to long
                images = np.copy(images)  # Make a copy to ensure it is writable
                images = torch.from_numpy(images)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                total_val += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / total_val)
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss / total_train:.4f}, Val Loss: {val_loss / total_val:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Check if the current validation accuracy is the best we've seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy  # Update the best known accuracy
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model
            print("Saved new best model")
            non_improve_counter = 0  # Reset counter
        else:
            non_improve_counter += 1  # Increment counter if no improvement

    return train_losses, val_losses, train_accuracies, val_accuracies

if __name__ ==  "__main__":
    ## 2. Data Augmentation for Car Racing
    df = pd.read_excel(
        r"C:\Users\Admin\Desktop\fau\second semester\ml lab\assigment 2\Imitation-Learning-Autonomous-Driving-Agent\action_snapshots.xlsx")

    image_paths = df['Snapshot'].values
    labels = df['Action'].apply(eval).values
    discrete_labels = np.array([transform_label(label) for label in labels])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming 'discrete_labels' contains all your labels for the dataset
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(discrete_labels), y=discrete_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    #image_paths = [str(path) for path in image_paths]
    #image_paths = [os.path.join(base_path, path) for path in image_paths]

    # Define the data augmentations
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Calculate splits
    train_size = int(0.6 * len(image_paths))
    val_size = int(0.2 * len(image_paths))
    test_size = len(image_paths) - train_size - val_size
    test_paths = image_paths[0:test_size]
    val_paths = image_paths[test_size:test_size+val_size]
    training_paths = image_paths[test_size + val_size:]

    # Create the dataset and data loader
    train_data = CustomDataset(image_paths=training_paths, labels=discrete_labels[test_size + val_size:], transform=None)
    val_data = CustomDataset(image_paths = val_paths, labels=discrete_labels[test_size:test_size+val_size], transform=None)
    test_data = CustomDataset(image_paths = test_paths, labels=discrete_labels[0:test_size], transform=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=20, patience=5)

    # Plotting the figures

    # Figure one: Loss

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Figure two: Accuracy

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Test
    test_loss = 0
    total_test = 0
    correct_test = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            labels = labels.long()  # Convert labels to long
            images = np.copy(images)  # Make a copy to ensure it is writable
            images = torch.from_numpy(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            total_test += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = correct_test / total_test
    print(f'Test Loss: {test_loss / total_test:.4f}, Test Accuracy: {test_accuracy:.4f}')

    predicted_np = predicted.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_np, predicted_np)
    np.set_printoptions(precision=2)
    class_names = ["nothing", "left", "right","gas", "break"]

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt='g', cbar=False, cmap='Blues');

    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # Add tickmarks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.show()



