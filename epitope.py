import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from quantiprot.metrics import aaindex
from quantiprot.utils.io import load_fasta_file
from sklearn.model_selection import train_test_split


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


def compute_feature_matrix(protein: str) -> pd.DataFrame:
    """
    Given a string of amino acids, compute all features for each one.
    :param protein:
    :return: a len(protein) x 6 matrix
    """
    df_result = pd.DataFrame(
        np.zeros((len(protein), NUM_FEATURES)),
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


def df_type_of_amino_acid(protein: str) -> pd.Series:
    """
    Compute type of each amino acid. Consider B, J, X, Z as separate amino acids.
    :param protein: string
    """
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


def compute_mapping_according_to_dict(mapping: dict, protein: str) -> list:
    """
    Given a mapping, return mapping[letter] for every letter in protein. B, J, X, Z values will be the average of the amino acids they represent.
    :param mapping: a dictionary of a value for each amino acid
    :param protein: string
    :return: a list of mapping[letter] for every letter in protein.
    """
    mapping["B"] = (mapping["D"] + mapping["N"]) / 2
    mapping["J"] = (mapping["I"] + mapping["L"]) / 2
    mapping["X"] = sum(list(mapping.values())) / len(mapping)
    mapping["Z"] = (mapping["E"] + mapping["Q"]) / 2
    return [mapping[amino_acid] for amino_acid in protein]


def create_dataset(protein_df: pd.DataFrame) -> tuple:
    """
    :param protein_df: Given a len(protein) x 6 feature matrix
    :return: A tuple of two lists: a 9x6 matrix list and a label list.
    """
    x = []
    Y = []
    for protein in protein_df["protein"]:
        feature_matrix = compute_feature_matrix(protein.upper()).to_numpy()
        for i in range(4, len(protein) - 4):
            x.append(feature_matrix[i - 4 : i + 5])
            Y.append(float(protein[i].isupper()))
    return x, Y


def get_train_test_validation(protein_filename: str) -> tuple:
    """
    The function divides the data into three datasets
    :param protein_filename: Given the file name to open the fasta file
    :return: Three datasets - Train, Test and Validation
    """
    protein_file = load_fasta_file(protein_filename)
    proteins = ["".join(fasta_seq.data) for fasta_seq in protein_file]
    train, test = train_test_split(
        pd.DataFrame(data={"protein": proteins}), train_size=0.8
    )
    test, validation = train_test_split(test, train_size=0.5)
    return train, test, validation


def calculate_ss(protein: str, window_size: int) -> pd.Series:
    """
    The function calculates the secondary structure feature
    :param protein: string
    :param window_size: int, not the same window as the one given to the matrix
    :return: pd.Series of secondary structure
    """
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
