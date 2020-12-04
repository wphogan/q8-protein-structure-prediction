import argparse
import os

import torch.nn as nn
import torch.optim as optim

from train_utils import *

np.random.seed(42)
parser = argparse.ArgumentParser(description='Choose a config file')

# experiment
parser.add_argument(
    '--experiment',
    default='dummy',
    help='Choose a config file (default: \'base\')'
)

args = parser.parse_args()

# grab values from arguments
experiment = args.experiment
from models.base_model import BaseModel as Model


from configs import configs as cfg
cfg = cfg.cfg[experiment]

batch_size = cfg["batch_size"]
valid_batch_size = cfg["valid_batch_size"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
lr = cfg["lr"]

cb513_path = cfg["cb513"]
model_type = experiment


if __name__ == "__main__":

    data = json.load(open(cb513_path, "r"))
    ids = np.arange(len(data))
    
    best_model_path = os.path.join("models", model_type, "best_model.pt")
    stats_path = os.path.join("stats", model_type, "stats.pkl")

    if os.path.exists(best_model_path):
        print("Model exists. Loading from {0}".format(best_model_path))
        model = torch.load(best_model_path)
    else:
        print("NO BEST MODEL")
        print("Exiting...")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model is using GPU: {0}".format(next(model.parameters()).is_cuda))
    
    test_loader, len_test = get_loader(protein_data=data,
                                       ids=ids,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)

    acc = test(model, test_loader, device)

    print(acc)

    with open(stats_path, "rb") as f:
        stats_dict = pkl.load(f)

    stats_dict["test"]["acc"] = acc

    with open(stats_path, "wb") as f:
        pkl.dump(stats_dict, f)
