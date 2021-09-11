import torch
from torch import nn
import torch.nn.functional as F

VOLUME_INDEX = 0
HYDROPHOBICITY_INDEX = 1
POLARITY_INDEX = 2
RSA_INDEX = 3
SS_INDEX = 4
TYPE_INDEX = 5


def one_hot_type_and_ss(batch: torch.tensor) -> list:
    """
    Change the type and ss from one column of categoirical data to one hot columns.
    """
    new_batch = []
    for feature_matrix in batch:
        ss_one_hot = F.one_hot(
            feature_matrix[:, SS_INDEX].to(torch.int64), num_classes=4
        )
        type_one_hot = F.one_hot(
            feature_matrix[:, TYPE_INDEX].to(torch.int64), num_classes=24
        )
        feature_matrix = feature_matrix[:, 0:SS_INDEX]
        new_feature_matrix = torch.cat((feature_matrix, ss_one_hot, type_one_hot), 1)
        new_batch.append(new_feature_matrix)
    return new_batch


def normalize_column(matrix: torch.tensor, index: int):
    """
    Min-max normalize columns index of matrix.
    """
    col = matrix[:, index]
    col_normalized = col - col.min(0, keepdim=True)[0]
    col_normalized /= col.max(0, keepdim=True)[0]
    matrix[:, index] = col_normalized
    return matrix


def preprocess_batch(batch: torch.tensor) -> torch.tensor:
    """
    One hot encode and normalize certain columns of the feature matrix.
    """
    batch = one_hot_type_and_ss(batch)
    batch = [normalize_column(mat, HYDROPHOBICITY_INDEX) for mat in batch]
    batch = [normalize_column(mat, POLARITY_INDEX) for mat in batch]
    batch = [normalize_column(mat, RSA_INDEX) for mat in batch]
    return torch.stack(batch)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=15, kernel_size=3)
        self.fc1 = nn.Linear(in_features=2100, out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = preprocess_batch(x)
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.batch_norm(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x
