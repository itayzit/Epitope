import pickle

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
TRAIN_DATA_SET = "train_data_set"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
BATCH_SIZE = 16


def get_dataset(under_sampling: bool) -> TensorDataset:
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

    return TensorDataset(x_ten, Y_ten)


# We'll use this code to use the train_data_loader in memory
train_data_loader = data.DataLoader(
    get_dataset(True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

test_data_loader = data.DataLoader(
    get_dataset(False),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


# We'll use this code to write to write the dataset to a file and upload it to google collab.
def write_to_file(data_set, filename):
    with open(filename, "wb") as outfile:
        pickle.dump(data_set, outfile)
    print(f"finished writing to {filename}")


write_to_file(get_dataset(True), TRAIN_DATA_SET)

with open(TRAIN_DATA_SET, "rb") as f:
    train_dataset = pickle.load(f)

# works great until this line!

classes = ("epitope", "not_epitope")
#%%


# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=3, padding=1
        )  # 6 kernels of size 5x5, c_in=1 because we have one channel (not like RGB)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3)
        # TODO: calculate the output size (H_out) from the equation under "shape" in https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3)
        self.fc1 = nn.Linear(
            in_features=90, out_features=256
        )  # use the result of the second H_out to determine 16 * 5 * ?
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # out: (16, 6, 9, 6)
        x = F.relu(self.conv2(x))  # out: (16, 3, 7, 4)
        x = F.relu(self.conv3(x))  # out: (16, 9, 5, 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# Define loss function and optimizer:
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

predict = lambda y: torch.argmax(torch.softmax(y, dim=1))


def get_val(val_loader):
    net.eval()
    val_loss = 0.0
    true_positives = 0
    all_samples = 0
    with torch.no_grad():
        for data, labels in val_loader:
            all_samples += labels.shape[0]
            y = net(data)
            predictions = predict(y)
            val_loss += criterion(y, labels.long()).item()
            true_positives += torch.sum((predictions == labels))
    return val_loss / all_samples, true_positives / all_samples


#%%
# Train the network
EPOCHS = 10
net.train()
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_loss += running_loss
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    val_loss, tp_rate = get_val(test_data_loader)
    print("*" * 10, "END OF EPOCH {}", "*" * 10).format(epoch)
    print(
        "avg train loss: {:.4f}\tavg val loss: {:.4f}\ttp rate: {}".format(
            epoch_loss / len(train_data_loader.dataset), val_loss, tp_rate
        )
    )

print("Finished Training")
