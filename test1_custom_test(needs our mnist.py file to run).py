from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision
from data import df_train, df_test
from model1 import rbf_prototypes, LeNet5


def test(dataloader,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmin(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad = torchvision.transforms.Pad(2, fill=0, padding_mode='constant')
    mnist_test = mnist.MNISTDataset(df_test)
    test_dataloader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    model = LeNet5(rbf_prototypes).to(device)
    model.load_state_dict(torch.load("LeNet5_1.pth", map_location=device))
    test(test_dataloader, model)

# Main function
if __name__=="__main__":
    main()
