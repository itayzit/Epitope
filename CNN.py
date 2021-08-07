import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

#%%
import epitope

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
BATCH_SIZE = 4


trainloader = data.DataLoader(
    tuple(epitope.trainset(epitope.get_train_and_test()[0])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

classes = ("Epitope", "NotEpitope")
#%%


# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, epitope.NUM_FEATURES, len(epitope.AMINO_ACIDS))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)


net = Net()


# Define loss function and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
