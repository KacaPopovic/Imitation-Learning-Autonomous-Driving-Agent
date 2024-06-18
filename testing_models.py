from imitation_learning import *
from model import *
import numpy as np
import torch
import cv2
def create_video(dataloader, output_file = 'output.mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first_sample = next(iter(dataloader))[0]
    first_sample = np.transpose(first_sample, (0,2,3,1))  # Pytorch has channels as first dimension
    height, width, channels = first_sample.shape[1], first_sample.shape[2], first_sample.shape[3]
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))

    for i,data in enumerate(dataloader):
        inputs,_ = data
        inputs = inputs.numpy()
        images = np.transpose(inputs, (0, 2, 3, 1))
        images = (images*255).astype(np.uint8)
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out.write(image)
    out.release()

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def test_model(model, test_loader, criterion, mode):
    model.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        test_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for images, labels in test_loader:
            labels = labels.long()  # Convert labels to long
            images = np.copy(images)  # Make a copy to ensure it is writable
            images = torch.from_numpy(images)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            total_test += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()
    test_accuracy = correct_test / total_test
    if mode == 1:
        print("Test results with no augmentations")
    elif mode == 4:
        print("Test results with upside-down flip")
    elif mode == 5:
        print("Test results with vertical flip")

    elif mode == 6:
        print("Test results with brown street")
    print(f'Test Loss: {test_loss / total_test}')
    print(f'Test Accuracy: {test_accuracy:.4f}')


def main():
   model = load_model('best_model1.pth')
   criterion = nn.CrossEntropyLoss()
   path = r"C:\Users\Admin\Desktop\fau\second semester\ml lab\assigment 2\Imitation-Learning-Autonomous-Driving-Agent\action_snapshots.xlsx"

   # Variable mode defines what type of transformations are applied to the datasets
   # Zero case - mode 1 - no augmentations on any dataset

   #train_data, val_data, test_data = load_data(path, mode = 1)
   #test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
   #test_model(model, test_dataloader, criterion, mode = 1)
   #create_video(test_dataloader)

   # First case - mode 4 - 1. upside-down frame flip,

   #model1 = load_model('best_model1.pth')
   #train_data1, val_data1, test_data1 = load_data(path, mode=4)
   #test_dataloader1 = DataLoader(test_data1, batch_size=128, shuffle=False)
   #test_model(model1, test_dataloader1, criterion, mode = 4)
   #create_video(test_dataloader1, output_file= "output1.mp4")

   # Second case - mode 5 -left-right frame flip

   #model2 = load_model('best_model1.pth')
   #train_data2, val_data2, test_data2 = load_data(path, mode=5)
   #test_dataloader2 = DataLoader(test_data2, batch_size=128, shuffle=False)
   #test_model(model2, test_dataloader2, criterion, mode = 5)
   #create_video(test_dataloader2, output_file="output2.mp4")

   # Third case - mode 6 brown street colour
   model3 = load_model('best_model1.pth')
   train_data3, val_data3, test_data3 = load_data(path, mode=6)
   test_dataloader3 = DataLoader(test_data3, batch_size=128, shuffle=False)
   test_model(model3, test_dataloader3, criterion, mode=6)
   create_video(test_dataloader3, output_file="output3.mp4")



if __name__ == '__main__':
    main()
