"""All of these functions were used in the google colab notebook in order to calculate values we used later on.
To test a function, copy and paste it to the google collab notebook."""
import torch
from torch import nn
from torch.utils import data
from sklearn.metrics import roc_curve
import numpy as np

device = torch.device("cuda")


def find_optimal_threshold(net: nn.Module, test_data_loader: data.DataLoader) -> float:
    """
    Find the optimal threshold according to the roc curve based on the test dataset.
    :param net: The CNN model.
    :param test_data_loader:
    :return: Float of the optimal threshold.
    """
    all_probs = []
    with torch.no_grad():
        for d, _ in test_data_loader:
            y = net(d.to(device)).squeeze(1)
            all_probs.append(y)
    fpr, tpr, thresholds = roc_curve(
        test_data_loader.dataset.tensors[1].cpu().numpy(),
        torch.cat(all_probs).cpu().numpy(),
    )
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print("Best Threshold=%f, G-Mean=%.3f" % (thresholds[ix], gmeans[ix]))
    return thresholds[ix]


def get_acc(net, data_loader):
    net.eval()
    true_predictions = 0
    with torch.no_grad():
        for d, labels in data_loader:
            y = net(d.to(device)).squeeze(1)
            predictions = torch.round(y)
            true_predictions += torch.sum(predictions == labels.to(device)).item()
    return true_predictions / (len(data_loader.dataset))
