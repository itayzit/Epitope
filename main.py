import argparse
from typing import Optional
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
import epitope
import network

OPTIMAL_THRESHOLD = 0.43808767  # Computed using utils.find_optimal_threshold

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-p",
    dest="protein",
    required=False,
)


def _load_net():
    net = network.Net()
    net.load_state_dict(
        torch.load("network_state_dict.pickle", map_location=torch.device("cpu"))
    )
    net.eval()
    return net


def main(protein: Optional[str]):
    """
    Given a string that represents a protein, return string with uppercase letters where the the network predicts the amino acid to be part of the epitope.
    :param protein: A string that represents a protein
    :return: string with uppercase letters where the epitope is.
    """
    if not protein:
        protein = arg_parser.parse_args().protein
    protein = protein.lower()
    net = _load_net()
    protein_pred = protein[:4]
    x, _ = epitope.create_dataset(pd.DataFrame(data={"protein": [protein]}))
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    data_loader = iter(data.DataLoader(dataset, batch_size=1, shuffle=False))
    all_predictions = []
    with torch.no_grad():
        for i in range(4, len(data_loader) + 4):
            d = next(data_loader)
            prediction = net(d[0]).squeeze(1)[0]
            all_predictions.append(prediction)
            if prediction - OPTIMAL_THRESHOLD >= 0.0:
                protein_pred += protein[i].upper()
            else:
                protein_pred += protein[i].lower()

    protein_pred += protein[-4:]
    with open("output.txt", "w") as out:
        out.write(f"Prediction for protein={protein} is:\n{protein_pred}\n\n")
        for i in range(len(all_predictions)):
            out.write(
                f"probability for acid={protein[i + 4]} is {all_predictions[i]}\n"
            )
    print(f"prediction is {protein_pred}, go to output.txt for more info")
    return protein_pred


if __name__ == "__main__":
    main(None)
