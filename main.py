import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import TensorDataset
import epitope
from sklearn.metrics import roc_curve

OPTIMAL_THRESHOLD = 0.5


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
            y = net(d.to(torch.cuda)).squeeze(1)
            all_probs.append(y)
    fpr, tpr, thresholds = roc_curve(
        test_data_loader.dataset.tensors[1].cpu().numpy(),
        torch.cat(all_probs).cpu().numpy(),
    )
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print("Best Threshold=%f, G-Mean=%.3f" % (thresholds[ix], gmeans[ix]))
    return thresholds[ix]


def round_to_threshold(num: float) -> float:
    if num - OPTIMAL_THRESHOLD >= 0.0:
        return 1.0
    return 0.0


# TODO: test it!
def main(protein: str):
    """
    Given a string that represents a protein, return string with uppercase letters where the the network predicts the amino acid to be part of the epitope.
    :param protein: A string that represents a protein
    :return: string with uppercase letters where the epitope is.
    """
    net = torch.load("model")  # TODO
    protein_pred = protein[:4]
    x, _ = epitope.create_dataset(pd.DataFrame(data={"protein": [protein]}))
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    data_loader = iter(data.DataLoader(dataset, batch_size=1, shuffle=False))
    with torch.no_grad():
        for i in range(len(data_loader)):
            d = next(data_loader)
            y = net(d[0]).squeeze(1)
            prediction = round_to_threshold(y[0])
            if prediction == 0.0:
                protein_pred += protein[i + 4].lower()
            else:
                protein += protein[i + 4].upper()
    return protein_pred + protein[-4:]
