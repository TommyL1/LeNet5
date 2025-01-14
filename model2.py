import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data import df_train, df_test

os.environ['NO_MACOS_IMK_WARNINGS'] = '1'

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(15),  

    transforms.RandomAffine(0, translate=(0.1, 0.1)), 
    transforms.RandomHorizontalFlip(),  
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
])


class MNISTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale

        # Resize the image to 32x32
        img = img.resize((32, 32), Image.BICUBIC)

        # Apply data augmentation
        if self.transform:
            img = self.transform(img)

        # Convert to tensor
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float).view(32, 32)
        img_tensor /= 255.0  # Normalize pixel values to [0, 1]
        img_tensor = 1.0 - img_tensor  # Invert colors (MNIST convention)

        # Add channel dimension for CNN
        img_tensor = img_tensor.unsqueeze(0)

        label = int(row['label'])
        return img_tensor, label



train_dataset = MNISTDataset(df_train, transform=train_transform)  # Use training transformations
test_dataset = MNISTDataset(df_test, transform=test_transform)    # Use test transformations

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class ScaledTanh(nn.Module):
    def __init__(self, A=1.7159, S=2.0/3.0):
        super(ScaledTanh, self).__init__()
        self.A = A
        self.S = S

    def forward(self, x):
        return self.A * torch.tanh(self.S * x)

class SubsamplingLayer(nn.Module):
    def __init__(self, n_maps):
        super(SubsamplingLayer, self).__init__()
        self.a = nn.Parameter(torch.ones(n_maps) * 0.5)
        self.b = nn.Parameter(torch.zeros(n_maps))
        self.activation = ScaledTanh()

    def forward(self, x):
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        N, C, H, W = pooled.shape
        a = self.a.view(1, C, 1, 1)
        b = self.b.view(1, C, 1, 1)
        out = self.activation(a * pooled + b)
        return out

class RBF(nn.Module):
    def __init__(self, in_dim, prototypes):
        super(RBF, self).__init__()
        self.register_buffer('prototypes', prototypes)
        self.in_dim = in_dim
        self.num_classes = prototypes.shape[0]

    def forward(self, x):
        x_exp = x.unsqueeze(1)
        w_exp = self.prototypes.unsqueeze(0)
        dist = (x_exp - w_exp).pow(2).sum(dim=2)
        return dist

def load_rbf_prototypes(folder='digits_jpeg'):
    prototypes = []
    for digit in range(10):
        digit_dir = os.path.join(folder, str(digit))
        files = [f for f in os.listdir(digit_dir) if f.lower().endswith('.jpeg')]
        if not files:
            raise FileNotFoundError(f"No JPEG files found in {digit_dir}")

        img_path = os.path.join(digit_dir, files[0])
        img = Image.open(img_path).convert('L')
        img = img.resize((12, 7), Image.BICUBIC)

        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.uint8)
        img_tensor = img_tensor.view(7, 12).float() / 255.0

        bin_tensor = torch.where(img_tensor > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
        bin_tensor = bin_tensor.view(-1)
        prototypes.append(bin_tensor)

    prototypes = torch.stack(prototypes, dim=0)
    prototypes = (prototypes - prototypes.mean()) / prototypes.std()
    prototypes = prototypes * 1.7159
    return prototypes

def compute_loss(output_dist, targets, j=0.1):
    j_tensor = torch.tensor(j, device=output_dist.device)
    N = output_dist.size(0)
    correct_class_dist = output_dist[torch.arange(N), targets]
    exp_neg_dist = torch.exp(-output_dist)
    log_sum_exp = torch.log(torch.exp(-j_tensor) + exp_neg_dist.sum(dim=1))
    loss_per_sample = correct_class_dist + log_sum_exp
    return loss_per_sample.mean()

def compute_error_rate(model, data_loader, device):
    model.eval()
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            distances = model(images)
            preds = torch.argmin(distances, dim=1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
    return 1.0 - (num_correct / num_total)

class LeNet5Modified(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5Modified, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # Increased filters
        self.relu1 = nn.ReLU()  # Replace ScaledTanh with ReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Second conv layer
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Adjust for 32x32 input size
        self.relu_fc1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 84)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Softmax for output probabilities


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)  # Apply dropout during training
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rbf_prototypes = load_rbf_prototypes('digits_jpeg')
model = LeNet5Modified(num_classes=10).to(device)

class ConstantStepOptimizer:
    def __init__(self, model, step_size=0.001):
        self.model = model
        self.step_size = step_size

    def step(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param -= self.step_size * param.grad

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

constant_optimizer = ConstantStepOptimizer(model, step_size=0.001)

num_epochs = 20
train_error_rates = []
test_error_rates = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        distances = model(images)
        loss = compute_loss(distances, labels, j=0.1)

        loss.backward()
        constant_optimizer.step()
        constant_optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches
    train_error = compute_error_rate(model, train_loader, device)
    test_error = compute_error_rate(model, test_loader, device)

    train_error_rates.append(train_error)
    test_error_rates.append(test_error)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Train Error: {train_error * 100:.2f}%, Test Error: {test_error * 100:.2f}%")

# Plotting and visualization
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_error_rates) + 1), [rate * 100 for rate in train_error_rates], label='Train Error')
plt.plot(range(1, len(test_error_rates) + 1), [rate * 100 for rate in test_error_rates], label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Error Rate (%)')
plt.title('Training and Testing Error Rates Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "LeNet5_2.pth")
print("Model saved as LeNet5_2.pth")

def compute_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            distances = model(images)
            preds = torch.argmin(distances, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(10))
    return cm

def visualize_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def find_most_confusing_examples(model, data_loader, device):
    model.eval()
    most_confusing = {}
    highest_confidences = {}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            distances = model(images)
            confidences, preds = distances.min(dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = preds[i].item()
                confidence = confidences[i].item()

                if true_label != predicted_label:
                    if true_label not in most_confusing or confidence < highest_confidences[true_label]:
                        most_confusing[true_label] = (images[i].cpu().numpy(), predicted_label, confidence)
                        highest_confidences[true_label] = confidence

        for true_label, (img, pred_label, conf) in most_confusing.items():
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"True: {true_label}, Pred: {pred_label}, Confidence: {conf:.2f}")
            plt.show()

conf_matrix = compute_confusion_matrix(model, test_loader, device)
visualize_confusion_matrix(conf_matrix)
find_most_confusing_examples(model, test_loader, device)