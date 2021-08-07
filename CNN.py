import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import torch.nn.functional as F
import epitope

#%%

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
BATCH_SIZE = 4


trainloader = data.DataLoader(
    epitope.trainset(epitope.get_train_and_test()[0]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

# works great until this line!

classes = ("epitope", "not_epitope")
#%%


# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 6, 5
        )  # 6 kernels of size 5x5, c_in=1 because we have one channel (not like RGB)
        # TODO: calculate the output size (H_out) from the equation under "shape" in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.pool = nn.MaxPool2d(2, 2)  # divide by 2
        self.conv2 = nn.Conv2d(
            6, 16, 5
        )  # use the result of H_out as the c_in of this layer
        # TODO: calculate the output size (H_out) from the equation under "shape" in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.fc1 = nn.Linear(
            16 * 5 * 5, 120
        )  # use the result of the second H_out to determine 16 * 5 * ?
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# Define loss function and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#%%
# Train the network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")
