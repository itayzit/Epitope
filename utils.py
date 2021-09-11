"""All of these functions were used in the google colab notebook in order to calculate values we used later on.
To test a function, copy and paste it to the google collab notebook."""
import torch
from quantiprot.utils.io import load_fasta_file
from torch import nn
from torch.utils import data
from sklearn.metrics import roc_curve
import numpy as np
import main

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


def get_data_loader_acc(net, data_loader):
    net.eval()
    true_predictions = 0
    with torch.no_grad():
        for d, labels in data_loader:
            y = net(d.to(device)).squeeze(1)
            predictions = torch.round(y)
            true_predictions += torch.sum(predictions == labels.to(device)).item()
    return true_predictions / (len(data_loader.dataset))


class ProteinAccuracy:
    def __init__(self, protein, acc):
        self.protein = protein
        self.acc = acc


def protein_prediction_accuracy(protein, protein_pred):
    trues = 0
    for i in range(4, len(protein) - 4):
        if (protein[i].isupper() and protein_pred[i].isupper()) or (
            protein[i].islower() and protein_pred[i].islower()
        ):
            trues += 1
    return trues / (len(protein) - 8)


def find_5_best_proteins(filename):
    protein_file = load_fasta_file(filename)
    proteins = ["".join(fasta_seq.data) for fasta_seq in protein_file]
    protein_accs = []
    j = 0
    for protein in proteins:
        print(f"** now handling protein number {j} out of {len(proteins)}**")
        protein_accs.append(
            ProteinAccuracy(
                protein, protein_prediction_accuracy(protein, main.main(protein))
            )
        )
        j += 1
    protein_accs.sort(key=lambda p: p.acc, reverse=True)
    return [p.protein for p in protein_accs]
