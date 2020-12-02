import time
import os
import numpy as np
import torch
import pickle as pkl
from collections import defaultdict

from utils import *


def rec_dd():
    return defaultdict(rec_dd)


def train(epochs, model, stats_path,
          train_loader, val_loader,
          optimizer, criterion,
          len_train, len_val,
          latest_model_path,
          best_model_path, optim_path, device, early_stop=10):
    
    fmt_string = "Epoch[{0}/{1}], Batch[{2}/{3}], Train Loss: {4}"

    # Load stats if path exists
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats_dict = pkl.load(f)
        print(stats_dict["best_epoch"])
        start_epoch = stats_dict["next_epoch"]
        min_val_loss = stats_dict["valid"][stats_dict["best_epoch"]]["loss"]
        print("Stats exist. Loading from {0}. Starting from Epoch {1}".format(stats_path, start_epoch))
    else:
        min_val_loss = np.inf
        stats_dict = rec_dd()
        start_epoch = 0

        # See loss before training
        val_loss = val(-1, model, val_loader, len_val, criterion, epochs, device)

        # Update statistics dict
        stats_dict["valid"][-1]["loss"] = val_loss

    model.train()
    for epoch in range(start_epoch, epochs):
        train_loss = 0.
        all_labels = []
        all_predictions = []

        ts = time.time()
        for iter, (X, Y, seq_lens) in enumerate(train_loader):
            optimizer.zero_grad()

            X = X.permute([0, 2, 1]).long().to(device)
            Y = Y.to(device)

            outputs = model(X, Y)
            
            loss = 0
            for y, t, seq_len in zip(outputs, Y, seq_lens):
                y_cut = y[:seq_len]
                t_cut = t[:seq_len]
              
                loss += criterion(y_cut, t_cut)

            train_loss += loss.item()
            
            if iter % 10 == 0:
                print(fmt_string.format(epoch, epochs, iter, len(train_loader), loss.item()))

            loss.backward()
            optimizer.step()

        print("\nFinished Epoch {}, Time elapsed: {}, Loss: {}".format(epoch, time.time() - ts,
                                                                       train_loss / len(train_loader)))

        # Avg train loss. Batch losses were un-averaged before when added to train_loss
        stats_dict["train"][epoch]["loss"] = train_loss / len(train_loader)

        # The validation stats after additional epoch
        val_loss = val(epoch, model, val_loader, len_val, criterion, epochs, device)

        # Update statistics dict
        stats_dict["valid"][epoch]["loss"] = val_loss
        stats_dict["next_epoch"] = epoch + 1

        # Save latest model
        torch.save(model, latest_model_path)

        # Save optimizer state dict
        optim_state = {'optimizer': optimizer.state_dict()}
        torch.save(optim_state, optim_path)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            # Save best model
            torch.save(model, best_model_path)
            stats_dict["best_epoch"] = epoch
        else:
            early_stop -= 1

        # Save stats
        with open(stats_path, "wb") as f:
            pkl.dump(stats_dict, f)

        # Set back to train mode
        model.train()

    return stats_dict, model


def val(epoch, model, val_loader, len_val, criterion, epochs, device):

    fmt_string = "Epoch[{0}/{1}], Batch[{3}/{4}], Batch Validation Loss: {2}"
    all_predictions = []
    model.eval()
    num_val_batches = len(val_loader)
    loss = 0.
    with torch.no_grad():
        for iter, (X, Y, seq_lens) in enumerate(val_loader):
            batch_loss = 0

            X = X.permute([0, 2, 1]).long().to(device)
            Y = Y.to(device)
              
            outputs = model.gen(X)

            for y, t, seq_len in zip(outputs, Y, seq_lens):
                y_cut = y[:seq_len]
                t_cut = t[:seq_len]
               
                loss += criterion(y_cut, t_cut)
                
            batch_loss = loss.item()

            if iter % 10 == 0:
                print(fmt_string.format(epoch, epochs, batch_loss, iter, num_val_batches))

    # Avg loss
    loss /= num_val_batches
    print("Total Validation Loss: {0}".format(loss))

    return loss


def test(model, test_loader, device, experiment="Name of experiment"):
    all_predictions = []
    model.eval()
    loss = 0

    fmt_string = "Batch[{0}/{1}]"
    with torch.no_grad():
        for iter, (X, Y, seq_lens) in enumerate(test_loader):

            X = X.permute([0, 2, 1]).long().to(device)
            Y = Y.to(device)
              
            outputs = model.gen(X)

            for y, t, seq_len in zip(outputs, Y, seq_lens):
                y_cut = y[:seq_len]
                t_cut = t[:seq_len]
               
                loss += criterion(y_cut, t_cut).item()
            
              
            if iter % 10 == 0:
              print(fmt_string.format(iter, len(test_loader)))
              
    loss /= len(test_loader)
    return loss
              