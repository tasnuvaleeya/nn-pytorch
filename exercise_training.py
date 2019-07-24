import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

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
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

print("Initial weights - \n\n", model[0].weight)

criterion = nn.CrossEntropyLoss()
images, labels = next(iter(trainloader))

images.resize_(64, 784)
# Forward pass, then backward pass, then update weights
logps = model(images)
loss = criterion(logps, labels)
loss.backward()
print("Gradient - ", model[0].weight.grad)

# Take an update step and few the new weights

optimizer.step()
print("Updated weights - ", model[0].weight)
