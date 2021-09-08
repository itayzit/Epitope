import pandas as pd
import torch
from Bio import SeqIO
from quantiprot.utils.io import load_fasta_file
from torch.utils import data
from torch.utils.data import TensorDataset
import epitope

BATCH_SIZE = 16
OPTIMAL_THRESHOLD = 0.5


def create_dataset_from_fasta(file_name):
    protein_file = load_fasta_file(file_name + ".fasta")
    proteins = ["".join(fasta_seq.data) for fasta_seq in protein_file]
    x, _ = epitope.create_dataset(proteins)
    x_ten = torch.tensor(x, dtype=torch.float32)
    return TensorDataset(x_ten)


def round_to_threshold(ten: torch.Tensor):
    predictions = torch.zeros(ten.shape)
    for i, num in enumerate(ten):
        if num - OPTIMAL_THRESHOLD >= 0.0:
            predictions[i] = 1.0
        else:
            predictions[i] = 0.0
    return predictions


def pred_to_string(predictions, protein_str):
    pred_string = ""
    for i in range(len(predictions)):
        if predictions[i] == 0.0:
            pred_string += protein_str[i].lower()
        else:
            pred_string += protein_str[i].upper()
    return pred_string


def main(file_name):
    protein_file = load_fasta_file(file_name)
    proteins = ["".join(fasta_seq.data) for fasta_seq in protein_file]
    protein_predictions = []
    for protein in proteins:
        protein_pred = ""
        x, _ = epitope.create_dataset([protein])
        x_ten = torch.tensor(x, dtype=torch.float32)
        dataset = TensorDataset(x_ten)
        data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        net = torch.load("model")  # TODO
        with torch.no_grad():
            for i, d in enumerate(data_loader):
                y = net(d[0]).squeeze(1)
                predictions = round_to_threshold(y)
                protein_pred += pred_to_string(
                    predictions, protein[i + 4: i + 4 + y.shape[0]] # TODO: deal with the last one where the
                )
            protein_predictions.append(protein_pred)
