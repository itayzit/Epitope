import timeit

import pandas as pd
import numpy as np
from quantiprot.metrics import aaindex
from quantiprot.utils.io import load_fasta_file
from sklearn.model_selection import train_test_split
from Bio.SeqUtils.ProtParam import ProteinAnalysis

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


def split_to_windows(protein):
    # TODO
    return None


def compute_feature_matrix(protein):
    volume_mapping = compute_mapping_according_to_dict(
        aaindex.get_aa2volume().mapping, protein
    )
    hydrophobicity_mapping = compute_mapping_according_to_dict(
        aaindex.get_aa2mj().mapping, protein
    )
    polarity_mapping = compute_mapping_according_to_dict(
        aaindex.get_aaindex_file("GRAR740102").mapping, protein
    )
    rsa_mapping = compute_mapping_according_to_dict(RSA_DICT, protein)
    df_type = df_type_of_amino_acid(protein)
    df = pd.DataFrame(
        data={
            "volume": volume_mapping,
            "hydrophobicity": hydrophobicity_mapping,
            "polarity": polarity_mapping,
            "RSA": rsa_mapping,
        }
    )
    df = df.join(calculate_ss(protein, 10))
    df = df.join(df_type)
    assert len(df.columns) == 28
    return df.astype(float)


def df_type_of_amino_acid(protein):
    df = pd.DataFrame(np.zeros((len(protein), len(AMINO_ACIDS))), columns=AMINO_ACIDS)
    for i, amino_acid in enumerate(protein):
        if amino_acid == "B":
            df.loc[i] = [1 if (acid == "D" or acid == "N") else 0 for acid in AMINO_ACIDS]
        elif amino_acid == "J":
            df.loc[i] = [1 if (acid == "J" or acid == "L") else 0 for acid in AMINO_ACIDS]
        elif amino_acid == "X":
            df.loc[i] = [1 for _ in AMINO_ACIDS]
        elif amino_acid == "Z":
            df.loc[i] = [1 if (acid == "E" or acid == "Q") else 0 for acid in AMINO_ACIDS]
        else:
            df.loc[i] = [1 if amino_acid == acid else 0 for acid in AMINO_ACIDS]
    return df


def compute_mapping_according_to_dict(mapping, protein):
    mapping["B"] = (mapping["D"] + mapping["N"]) / 2
    mapping["J"] = (mapping["I"] + mapping["L"]) / 2
    mapping["X"] = sum(list(mapping.values())) / len(mapping)
    mapping["Z"] = (mapping["E"] + mapping["Q"]) / 2
    return [mapping[amino_acid] for amino_acid in protein]


def main():
    fasta_sequences = load_fasta_file("iedb_linear_epitopes.fasta")
    names = [fasta_seq.identifier for fasta_seq in fasta_sequences]
    proteins = ["".join(fasta_seq.data) for fasta_seq in fasta_sequences]
    train, test = train_test_split(
        pd.DataFrame(data={"name": names, "protein": proteins})
    )
    for protein in train["protein"]:
        feature_matrix = compute_feature_matrix(protein.upper())
        labels = [acid.isupper() for acid in protein]
        # TODO: train_main(feature_matrix, labels)


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
        # result.append(np.argmax(probabs))

    helix_result = [1 if result[i] == 0 else 0 for i in range(len(protein))]
    turn_result = [1 if result[i] == 1 else 0 for i in range(len(protein))]
    sheet_result = [1 if result[i] == 2 else 0 for i in range(len(protein))]
    other_structure_result = [1 if result[i] == 3 else 0 for i in range(len(protein))]
    return pd.DataFrame(data={"helix": helix_result, "turn": turn_result, "sheet": sheet_result, "other": other_structure_result})