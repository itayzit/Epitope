import timeit

import pandas as pd
import numpy as np
from quantiprot.metrics import aaindex
from quantiprot.utils.io import load_fasta_file
from sklearn.model_selection import train_test_split
from Bio.SeqUtils.ProtParam import ProteinAnalysis

PROTEIN_FILE = load_fasta_file("proteins_short.fasta")
# FILENAME = "iedb_linear_epitopes.fasta"

NUM_FEATURES = 6
WINDOW_SIZE = 9

AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
# https://www.sciencedirect.com/science/article/abs/pii/0022283676901911
RSA_DICT = {
    "A": 115,
    "D": 150,
    "C": 135,
    "E": 190,
    "F": 210,
    "G": 75,
    "H": 195,
    "I": 175,
    "K": 200,
    "L": 170,
    "M": 185,
    "N": 160,
    "P": 145,
    "Q": 180,
    "R": 225,
    "S": 115,
    "T": 140,
    "V": 155,
    "W": 255,
    "Y": 230,
}


def compute_feature_matrix(protein):
    df_result = pd.DataFrame(
        np.zeros((max_protein_len(), NUM_FEATURES)),
        columns=["volume", "hydrophobicity", "polarity", "RSA", "ss", "type"],
    )
    df_type = df_type_of_amino_acid(protein)
    df = pd.DataFrame(
        data={
            "volume": compute_mapping_according_to_dict(
                aaindex.get_aa2volume().mapping, protein
            ),
            "hydrophobicity": (
                compute_mapping_according_to_dict(aaindex.get_aa2mj().mapping, protein)
            ),
            "polarity": (
                compute_mapping_according_to_dict(
                    aaindex.get_aaindex_file("GRAR740102").mapping, protein
                )
            ),
            "RSA": (compute_mapping_according_to_dict(RSA_DICT, protein)),
        }
    )
    df = df.join(calculate_ss(protein, 10))
    df = df.join(df_type)
    assert len(df.columns) == NUM_FEATURES
    df_result.loc[0 : len(protein)] = df
    return df_result.astype(float)


def df_type_of_amino_acid(protein):
    d = {acid: float(i) for i, acid in enumerate(AMINO_ACIDS)}
    d.update(
        {
            "B": len(AMINO_ACIDS),
            "J": len(AMINO_ACIDS) + 1,
            "X": len(AMINO_ACIDS) + 2,
            "Z": len(AMINO_ACIDS) + 3,
        }
    )
    return pd.Series([d[acid] for acid in protein]).rename("type")


def compute_mapping_according_to_dict(mapping, protein):
    mapping["B"] = (mapping["D"] + mapping["N"]) / 2
    mapping["J"] = (mapping["I"] + mapping["L"]) / 2
    mapping["X"] = sum(list(mapping.values())) / len(mapping)
    mapping["Z"] = (mapping["E"] + mapping["Q"]) / 2
    return [mapping[amino_acid] for amino_acid in protein]


def trainset(train):
    train_set = []
    for protein in train["protein"]:
        train_set.append(
            (
                compute_feature_matrix(protein.upper()).to_numpy(),
                [acid.isupper() for acid in protein]
                + [0 for _ in range(MAX_PROTEIN_LEN - len(protein))],
            )
        )
    return train_set


def max_protein_len():
    return max([len(protein.data) for protein in PROTEIN_FILE])


MAX_PROTEIN_LEN = max_protein_len()


def get_train_and_test():
    fasta_sequences = PROTEIN_FILE
    names = [fasta_seq.identifier for fasta_seq in fasta_sequences]
    proteins = ["".join(fasta_seq.data) for fasta_seq in fasta_sequences]
    return train_test_split(pd.DataFrame(data={"name": names, "protein": proteins}))


def main():
    train, test = get_train_and_test()
    # CNN.train(trainset(train))


def calculate_ss(
    protein, window_size
):  # not the same window as the one given to the matrix
    result = [0 for _ in range(len(protein))]
    for i in range(len(protein)):
        start = max(0, i - (window_size // 2))
        end = min(len(protein), i + window_size // 2)
        sub_sequence = str(protein[start:end])
        probabs = list(ProteinAnalysis(sub_sequence).secondary_structure_fraction())
        probabs.append(1 - sum(probabs))
        if protein[i].lower() == "p":
            probabs[0] = 0.0
        result[i] = np.argmax(probabs)
    return pd.Series(result).rename("ss")
