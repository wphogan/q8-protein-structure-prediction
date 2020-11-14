import argparse
import pickle as pkl
from train_utils import *

np.random.seed(42)
parser = argparse.ArgumentParser(description='Choose a config file')

# experiment
parser.add_argument(
    '--experiment',
    default='dummy',
    help='Choose a config file (default: \'base\')'
)

# ensemble
parser.add_argument(
    '--num_models',
    default=1,
    help='Choose the number of models to ensemble (default: 1)'
)
args = parser.parse_args()

# grab values from arguments
experiment = args.experiment
num_models = int(args.num_models)

# base architecture
if "base" in experiment:
    from configs import base_config as cfg, dummy_config as cfg
else:
    experiment = "dummy"

# Get argument from config
cfg = cfg.cfg
batch_size = cfg["test_batch_size"]
valid_batch_size = cfg["valid_batch_size"]
num_workers = cfg["num_workers"]
epochs = cfg["epochs"]
lr = cfg["lr"]
num_features = cfg["num_features"]  # base=51, prot_vec=100
one_hot_embed = cfg["one_hot_embed"]
model_type = experiment

print(model_type)

if __name__ == "__main__":
    cb513_data = json.load(open("CB513.json", "r"))
    ids = np.arange(len(cb513_data))
    if experiment == "dummy":
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=[0, 1, 2],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           num_features=num_features)
    else:
        test_loader, len_test = get_loader(protein_data=cb513_data,
                                           ids=ids,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           num_features=num_features)

    if not os.path.exists(model_type):
        os.mkdir(model_type)

    models = []

    for model_num in range(1, num_models + 1):

        best_model_path = os.path.join("models", model_type + str(model_num), "best_model.pt")
        stats_path = os.path.join("stats", model_type + str(model_num), "stats.pkl")

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

        models.append(model)

    acc = test(model, test_loader, device, num_features, one_hot_embed)

    print(acc)

    with open(stats_path, "rb") as f:
        stats_dict = pkl.load(f)

    stats_dict["test"]["acc"] = acc

    with open(stats_path, "wb") as f:
        pkl.dump(stats_dict, f)
