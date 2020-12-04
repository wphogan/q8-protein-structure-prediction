import argparse
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import torch
from data_loader import get_loader

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

# base architecture
if "base" in experiment:
    from configs import base_config as cfg
    from models.S2S import S2S as Model
elif "bidi" in experiment:
    from configs import bidirectional_config as cfg
    from models.S2S import S2S_B as Model
elif "att" in experiment:
    from configs import att_config as cfg
    from models.S2S import S2S_att as Model
else:
    from configs import dummy_config as cfg
    from models.S2S import S2S as Model
    experiment = "dummy"

# Get argument from config
cfg = cfg.cfg
batch_size = cfg["batch_size"]
valid_batch_size = cfg["valid_batch_size"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
lr = cfg["lr"]
model_type = experiment

num_workers = cfg["num_workers"]
batch_size = cfg["batch_size"]

if __name__ == "__main__":
    # Should be path to TR6614 data
    data = json.load(open("../preprocess/TR6614_no_pssm_onedim.json", "r"))
    ids = np.arange(len(data))

    if experiment == "dummy":
        loader, _ = get_loader(protein_data=data,
                               ids=[0, 1, 2, 3, 5],
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)
    else:
        loader, _ = get_loader(protein_data=data,
                               ids=ids,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)


    latest_model_path = os.path.join("models", model_type, "latest_model.pt")

    if os.path.exists(latest_model_path):
        print("Model exists. Loading from {0}".format(latest_model_path))
        if torch.cuda.is_available():
            model = torch.load(latest_model_path)
            device = "cuda"
        else:
            model = torch.load(latest_model_path, map_location=torch.device('cpu'))
            device = "cpu"
    else:
        print("Model not trained")
        print("Exiting...")
        
    def gen(model, loader, device):
        fmt_string = "Batch[{0}/{1}]"

        gen_sequences = []
        model.eval()
        num_batches = len(loader)

        for iter, (X, _, _) in enumerate(loader):
            X = X.permute([0, 2, 1]).long().to(device)
            outputs = model.gen(X)

            print(fmt_string.format(iter, num_batches))
            gen_sequences.append(outputs.detach().cpu().numpy())


        return gen_sequences

    PSSM = gen(model, loader, device)
    PSSM = np.vstack(PSSM)
    one_dim_conservation = np.log(21) + np.sum(PSSM * np.log(PSSM + 1e-4), axis=2, keepdims=True)
    PSSM = np.concatenate([PSSM, one_dim_conservation], axis=2)
    
    if not os.path.exists("gen_seq"):
        os.mkdir("gen_seq")
    
    gen_save_path = os.path.join("gen_seq", "TR6614_{0}.npy".format(experiment))
    print(PSSM.shape)
    np.save(gen_save_path, PSSM)
