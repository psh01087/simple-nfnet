import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import NFNet
from optim import SGD_AGC


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])


# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = NFNet(num_classes=10, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5, activation='gelu').to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD_AGC(named_params=model.named_parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00002, nesterov=True)

# Train the model

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'nfnet.ckpt')