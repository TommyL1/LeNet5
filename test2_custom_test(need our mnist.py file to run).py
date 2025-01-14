from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from mnist import MNISTDataset2
from model2 import LeNet5Modified, rbf_prototypes, test_transform
from data import df_test  # Assuming df_test is correctly defined

def test(dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  

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
    test_dataset = MNISTDataset2(df_test, transform=test_transform)  
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5Modified(num_classes=10).to(device)
    model.load_state_dict(torch.load("LeNet5_2.pth", map_location=device))
    model.to(device)
    test(test_dataloader, model)

if __name__ == "__main__":
    main()
