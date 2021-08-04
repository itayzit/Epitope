import pandas as pd
import numpy as np
from quantiprot.metrics import aaindex
from quantiprot.utils.io import load_fasta_file
from sklearn.model_selection import train_test_split
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils.ProtParam import ProteinAnalysis

WINDOW_SIZE = 9


def split_to_windows(protein):
    # TODO
    return None


def _get_average_from_mapping(mapping):
    return sum(list(mapping.mapping.values())) / len(mapping.mapping)


def compute_feature_matrix(protein):
    volume_mapping = aaindex.get_aa2volume(default=_get_average_from_mapping(aaindex.get_aa2volume())) # use the average as default
    hydrophobicity_mapping = aaindex.get_aa2mj(default=_get_average_from_mapping(aaindex.get_aa2mj())) # use the average as default
    polarity_mapping = aaindex.get_aaindex_file('GRAR740102', default=_get_average_from_mapping(aaindex.get_aaindex_file('GRAR740102'))) # use the average as default
    # TODO: compute rsa
    # TODO: compute ss
    return pd.DataFrame(data={"volume": volume_mapping(protein), "hydrophobicity": hydrophobicity_mapping(protein), "polarity": polarity_mapping(protein), "type": list(protein)})


def main():
    fasta_sequences = load_fasta_file("iedb_linear_epitopes.fasta")
    names = [fasta_seq.identifier for fasta_seq in fasta_sequences]
    proteins = ["".join(fasta_seq.data) for fasta_seq in fasta_sequences]
    train, test = train_test_split(pd.DataFrame(data={"name": names, "protein": proteins}))
    for protein in train['protein']:
        feature_matrix = compute_feature_matrix(protein.upper())
        feature_matrix["type"] = feature_matrix["type"].astype('category').cat.codes # convert from categorical to numeric
        labels = [acid.isupper() for acid in protein]
        # TODO: train_main(feature_matrix, labels)


def calculate_rsa():  # draft
    # not working: dssp gets .pdb and we have .fasta, need to somehow convert fasta to pdb
    p = PDBParser()
    structure = p.get_structure("rsa", "iedb_linear_epitopes.pdb-seqres")
    model = structure[0]
    dssp = DSSP(model, "iedb_linear_epitopes.pdb-seqres", dssp='mkdssp')
    a_key = list(dssp.keys())[2]
    print(dssp[a_key])


def calculate_ss(protein, window_size): # not the same window as the one given to the matrix
    result = []
    for i in range(len(protein)):
        start = max(0, i - (window_size // 2))
        end = min(len(protein), i + window_size // 2)
        sub_sequence = str(protein[start:end])
        probabs = list(ProteinAnalysis(sub_sequence).secondary_structure_fraction())
        probabs.append(1 - sum(probabs))
        if protein[i].lower() == "p":
            probabs[0] = 0.0
        result.append(np.argmax(probabs))

    helix_result = [1 if result[i] == 0 else 0 for i in range(len(protein))]
    turn_result = [1 if result[i] == 1 else 0 for i in range(len(protein))]
    sheet_result = [1 if result[i] == 2 else 0 for i in range(protein)]
    other_structure_result = [1 if result[i] == 3 else 0 for i in range(protein)]
    return pd.DataFrame(np.vstack((helix_result, turn_result, sheet_result, other_structure_result)))
