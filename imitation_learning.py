import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN, CustomDataset


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

# Function for training and validation

def train_and_validate(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=10, patience = 3):
    best_val_accuracy = 0.0
    train_losses, val_losses, train_accuracies, val_accuracies, test_losses, test_accuracies = [], [], [], [], [], []

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
        correct_test = 0
        total_test = 0
        test_loss = 0

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

            for images, labels in test_loader:
                labels = labels.long()  # Convert labels to long
                images = np.copy(images)  # Make a copy to ensure it is writable
                images = torch.from_numpy(images)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                total_test += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / total_val)
        val_accuracies.append(val_accuracy)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss / total_test)
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss / total_train:.4f}, Val Loss: {val_loss / total_val:.4f}, Test Loss: {test_loss/total_test}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Check if the current validation accuracy is the best we've seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy  # Update the best known accuracy
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model
            print("Saved new best model")
            non_improve_counter = 0  # Reset counter
        else:
            non_improve_counter += 1  # Increment counter if no improvement

    return train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies

def load_data(path, mode = 1):
    ## 2. Data Augmentation for Car Racing
    if mode == 2:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif mode == 3:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])

    df = pd.read_excel(path)
    image_paths = df['Snapshot'].values
    labels = df['Action'].apply(eval).values

    discrete_labels = np.array([transform_label(label) for label in labels])

    train_size = int(0.6 * len(image_paths))
    val_size = int(0.2 * len(image_paths))
    test_size = len(image_paths) - train_size - val_size

    test_paths = image_paths[0:test_size]
    val_paths = image_paths[test_size:test_size + val_size]
    training_paths = image_paths[test_size + val_size:]
    test_labels = discrete_labels[0:test_size]
    val_labels = discrete_labels[test_size:test_size + val_size]
    training_labels = discrete_labels[test_size + val_size:]

    # Create the dataset and data loader
    train_data = CustomDataset(image_paths=training_paths, labels=training_labels, mode=mode, transform=transform)
    val_data = CustomDataset(image_paths=val_paths, labels=val_labels, mode = mode, transform=transform)
    test_data = CustomDataset(image_paths=test_paths, labels=test_labels, mode = mode, transform=None)

    return train_data, val_data, test_data

if __name__ ==  "__main__":

    # RGB training with no augmentation - 1, with augm - 2, grayscale - 3

    mode = 1

    path =r"C:\Users\Admin\Desktop\fau\second semester\ml lab\assigment 2\Imitation-Learning-Autonomous-Driving-Agent\action_snapshots.xlsx"
    train_data, val_data, test_data = load_data(path, mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)


    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies = train_and_validate(model, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, num_epochs=1, patience=5)

    # Plotting the figures

    # Figure one: Loss

    plt.figure(figsize=(12, 5))
    if mode == 1:
        title = 'Loss and Accuracy - RGB without Augm'
    elif mode == 2:
        title = 'Loss and Accuracy - RGB with Augm'
    else:
        title = 'Loss and Accuracy - grayscale'
    plt.title(title)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label ='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Figure two: Accuracy

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label ='Test Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()






