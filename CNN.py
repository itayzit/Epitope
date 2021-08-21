import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils import data
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from imblearn.under_sampling import RandomUnderSampler
import epitope

#%%

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
BATCH_SIZE = 16


def get_train_data_loader(under_sampling: bool = True) -> data.DataLoader:
    x, Y = epitope.create_dataset(epitope.get_train_and_test()[0])
    x_ten, Y_ten = torch.tensor(x, dtype=torch.float32), torch.tensor(
        Y, dtype=torch.float32
    )
    assert x_ten.shape[1:] == (9, 6)
    if under_sampling:
        under_sampler = RandomUnderSampler(random_state=42)
        num_of_samples, nx, ny = x_ten.shape
        x_2d = x_ten.reshape(
            (num_of_samples, nx * ny)
        )  # Reshape from 3d to 2d because under_sampler works only with 2d
        x_2d_balanced, Y_balanced = under_sampler.fit_resample(x_2d, Y)
        x_ten = torch.tensor(x_2d_balanced.reshape(-1, nx, ny), dtype=torch.float32)
        Y_ten = torch.tensor(Y_balanced, dtype=torch.float32)

    dataset = TensorDataset(x_ten, Y_ten)
    return data.DataLoader(
        dataset,
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
            in_channels=1, out_channels=6, kernel_size=5
        )  # 6 kernels of size 5x5, c_in=1 because we have one channel (not like RGB)
        self.pool = nn.MaxPool2d(2, 2)  # divide by 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=4, kernel_size=5)
        # TODO: calculate the output size (H_out) from the equation under "shape" in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.fc1 = nn.Linear(
            in_features=16 * 5 * 5, out_features=120
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
# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
# print("Finished Training")
