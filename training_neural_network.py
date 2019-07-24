import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Build a feef forward network

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
criterion = nn.CrossEntropyLoss()

# Get our data

images, labels = next(iter(trainloader))

# Flatten image
images = images.view(images.shape[0], -1)

# Forward pass, get our log-probabilities
logits = model(images)

loss = criterion(logits, labels)

print("Before Backward pass:", model[0].weight.grad)
loss.backward()
print("after Backward pass:", model[0].weight.grad)


# Training the network!

from torch import optim

optimizer = optim.SGD(params=model.parameters(), lr=0.01)
print(optimizer)