import json

import matplotlib.pyplot as plt
import pandas as pd
from seqeval.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from data_loader import *

colors = ["midnightblue", "maroon", "darkgreen", "indigo", "black", "darkslateblue", "purple", "cyan"]
temps = [0.1, 0.2, 0.7, 1.0, 1.5, 2.0]


def json_fix_keys(x):
    if isinstance(x, dict):
        key = list(x.keys())[0]
        if key.isnumeric():
            return {int(k): v for k, v in x.items()}
        else:
            return {k: v for k, v in x.items()}

    return x


def plot_loss(train_losses, valid_losses, arch="base"):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    # Valid dataset has -1 epoch b/c we wanted to see stats before training
    X_valid = np.arange(-1, len(valid_losses) - 1)
    X_train = np.arange(len(train_losses))
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    axs.set_title("Loss over Epochs ({0})".format(arch))
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")

    axs.plot(X_train, train_losses, label="Train", color=colors[0])
    axs.plot(X_valid, valid_losses, label="Valid", color=colors[1])

    axs.legend(loc='best', fontsize='x-large')
    axs.set_ylim([0, 1.1 * max(np.max(train_losses), np.max(valid_losses))])

    path_name = os.path.join("plots", arch + "_loss.png")
    plt.savefig(path_name)
    # plt.show()


def format_stats(stats_dict, length=50):
    train_info = {"loss": [],
                  "acc": []}

    valid_info = {"loss": [stats_dict["valid"]['-1']["loss"]],
                  "acc": [stats_dict["valid"]['-1']["acc"]]}

    best_epoch = stats_dict["best_epoch"] - 1

    for epoch in range(min(len(stats_dict["train"]), length)):
        train_info["loss"].append(stats_dict["train"][epoch]["loss"])
        valid_info["loss"].append(stats_dict["valid"][str(epoch)]["loss"])
        train_info["acc"].append(stats_dict["train"][epoch]["acc"])
        valid_info["acc"].append(stats_dict["valid"][str(epoch)]["acc"])

    return {"train": train_info, "valid": valid_info}


def plot_acc(train_losses, valid_losses, arch="base"):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    # Valid dataset has -1 epoch b/c we wanted to see stats before training
    X_valid = np.arange(-1, len(valid_losses) - 1)
    X_train = np.arange(len(train_losses))
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    axs.set_title("Acc over Epochs ({0})".format(arch))
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Acc")

    axs.plot(X_train, train_losses, label="Train", color=colors[0])
    axs.plot(X_valid, valid_losses, label="Valid", color=colors[1])

    axs.legend(loc='best', fontsize='x-large')
    axs.set_ylim([0, 1.1 * max(np.max(train_losses), np.max(valid_losses))])

    path_name = os.path.join("plots", arch + "_acc.png")
    plt.savefig(path_name)
    # plt.show()


def precision_recall_f1(y_pred, y_true):
    scores = {'precision_micro': precision_score(y_true, y_pred, average='micro'),
              'recall_micro': recall_score(y_true, y_pred, average='micro'),
              'f1_micro': f1_score(y_true, y_pred, average='micro')}

    # Micro
    print("test.precision_micro: ", scores['precision_micro'])
    print("test.recall_micro: ", scores['recall_micro'])
    print("test.f1_micro", scores['f1_micro'])

    # Macro
    scores['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    scores['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    scores['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    print("test.precision_macro: ", scores['precision_macro'])
    print("test.recall_macro: ", scores['recall_macro'])
    print("test.f1_macro: ", scores['f1_macro'])

    # Weighted
    scores['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    scores['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    scores['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    print("test.precision_weighted: ", scores['precision_weighted'])
    print("test.recall_weighted: ", scores['recall_weighted'])
    print("test.f1_weighted: ", scores['f1_weighted'])


def class_report_conf_matrix(predictions, labels, experiment):
    id_to_label = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'pad']
    labels = [id_to_label[i] for i in labels.tolist()]
    predictions = [id_to_label[i] for i in predictions.tolist()]
    cl_report = classification_report(labels, predictions)
    conf_mat = annot_confusion_matrix(labels, predictions)
    print(f"Classification Report:\n {cl_report}")
    print(f"Confusion Matrix:\n {conf_mat}")

    # Save test output
    df = pd.DataFrame()
    df['labels'] = labels
    df['predictions'] = predictions
    df.to_csv('stats/' + experiment + '/test_output.tsv', sep='\t', encoding='utf-8')
    print("Sample output saved in: stats/" + experiment + "/test_output.tsv")


def annot_confusion_matrix(valid_tags, pred_tags):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


def print_stats(arch):
    with open("stats/{0}/stats.pkl".format(arch), "rb") as f:
        stats_dict = pkl.load(f)

    stats_dict = json.loads(json.dumps(stats_dict), object_hook=json_fix_keys)
    formatted_stats_dict = format_stats(stats_dict)
    plot_loss(formatted_stats_dict["train"]["loss"], formatted_stats_dict["valid"]["loss"], arch)
    plot_acc(formatted_stats_dict["train"]["acc"], formatted_stats_dict["valid"]["acc"], arch)
