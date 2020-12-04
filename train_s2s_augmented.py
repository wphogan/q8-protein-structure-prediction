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
tr5534_path = cfg["tr5534"]
tr6114_path = cfg["tr6614"]
cb513_path = cfg["cb513"]
model_type = experiment


if __name__ == "__main__":

    if "dummy" in experiment:
        tr5534_data = json.load(open(cb513_path, "r"))
        tr6614_data = json.load(open(cb513_path, "r"))
        cb513_data = json.load(open(cb513_path, "r"))
    else:
        print(os.listdir("preprocess"))
        tr6614_data = json.load(open(tr6114_path, "r"))
        print("6614 loaded")
        tr5534_data = json.load(open(tr5534_path, "r"))
        print("5534 loaded")
        cb513_data = json.load(open(cb513_path, "r"))

    print("data loaded")
    counter = 0
    offset = len(tr5534_data)
    data = {}

    for i in range(len(tr5534_data)):
        data[str(i)] = tr5534_data[str(i)]

    for i in range(len(tr6614_data)):
        data[str(i + offset)] = tr6614_data[str(i)]
        

    len_train = len(data)
    percent_train = .8

    train_start = 0
    train_end = int(len_train * percent_train)

    val_start = train_end
    val_end = len_train

    print(train_start, train_end)
    print(val_start, val_end)

    np.random.seed(42)
    ids = np.random.choice(len_train, len_train, replace=False)

    if experiment == "dummy":
        train_loader, len_train = get_loader(protein_data=data,
                                             ids=[0, 1, 2],
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

        val_loader, len_val = get_loader(protein_data=data,
                                         ids=[0, 1, 2],
                                         batch_size=valid_batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

    else:
        train_loader, len_train = get_loader(protein_data=data,
                                             ids=ids[train_start:train_end],
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

        val_loader, len_val = get_loader(protein_data=data,
                                         ids=ids[val_start:val_end],
                                         batch_size=valid_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

    print(len_train, len_val)

    if not os.path.exists(os.path.join("models", model_type)):
        os.makedirs(os.path.join("stats", model_type), exist_ok=True)
        os.makedirs(os.path.join("models", model_type), exist_ok=True)

    latest_model_path = os.path.join("models", model_type, "latest_model.pt")
    best_model_path = os.path.join("models", model_type, "best_model.pt")
    optim_path = os.path.join("models", model_type, "optim.pt")
    stats_path = os.path.join("stats", model_type, "stats.pkl")


    def init_weights(m):
        try:
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()
        except:
            pass


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=8)
    model.apply(init_weights)
    print(type(model))

    if os.path.exists(latest_model_path):
        print("Model exists. Loading from {0}".format(latest_model_path))
        model = torch.load(latest_model_path)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(optim_path):
        print("Optimizer state dict exists. Loading from {0}".format(optim_path))
        optim = torch.load(optim_path)
        optimizer.load_state_dict(optim['optimizer'])

    model.to(device)
    print("Model is using GPU: {0}".format(next(model.parameters()).is_cuda))

    print(model)
    stats_dict, model = train(epochs, model, stats_path, train_loader, val_loader, optimizer, criterion,
                              len_train, len_val, latest_model_path, best_model_path, optim_path, device)
