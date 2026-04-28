#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rui.torch.utils — local copy of training utilities."""

import os
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", save_best_only=True,
                 save_optimizer_state=False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_optimizer_state = save_optimizer_state
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_best(self, model, optimizer):
        if self.save_optimizer_state:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       self.filepath)
        else:
            torch.save(model.state_dict(), self.filepath)


def plotEpoch(history, metric="average_loss", save_path=None):
    epochs = range(len(history["average_train_loss"]))
    if metric == "accuracy":
        plt.plot(epochs, history["train_accuracy"], "k", label="Training acc")
        plt.plot(epochs, history["val_accuracy"], "b", label="Validation acc")
        plt.axvline(x=int(np.argmax(history["val_accuracy"])), color="0.5")
        plt.title("Training and Validation Accuracy")
        plt.legend()
    elif metric == "total_loss":
        plt.plot(epochs, history["total_train_loss"], "k", label="Training loss")
        plt.plot(epochs, history["total_val_loss"], "b", label="Validation loss")
        plt.axvline(x=int(np.argmin(history["total_val_loss"])), color="0.5")
        plt.title("Total Training and Validation Loss")
        plt.legend()
    else:
        plt.plot(epochs, history["average_train_loss"], "k", label="Training loss")
        plt.plot(epochs, history["average_val_loss"], "b", label="Validation loss")
        plt.axvline(x=int(np.argmin(history["average_val_loss"])), color="0.5")
        plt.title("Average Training and Validation Loss")
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def train(model, train_dataloader, val_dataloader, optimizer, scheduler=None,
          clip_grad=None, loss_fn=nn.CrossEntropyLoss(), callbacks=None,
          device=None, evaluation=True, n_epochs=1, n_batch_per_report=5,
          accuracy_fn=None, lr_history=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = next(iter(train_dataloader))
    y_shape = batch[-1].squeeze().shape
    if len(y_shape) == 1:
        n_labels_per_instance = 1
    elif len(y_shape) == 2:
        n_labels_per_instance = y_shape[1]
    elif len(y_shape) == 3:
        n_labels_per_instance = y_shape[1] * y_shape[2]
    else:
        print(f"Error: Target tensors with {len(y_shape)} or more axes not supported!")

    history = {
        "total_train_loss": [], "average_train_loss": [], "train_accuracy": [],
        "total_val_loss": [],   "average_val_loss": [],   "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_state = None
    optimizer_state = None

    print(f"Start training on device {device}.\n")
    for epoch in range(1, n_epochs + 1):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | "
              f"{'Train Acc':^9} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 80)

        model.train()
        t0_epoch = t0_batch = time.time()
        total_correct = total_loss = total_abs_err = weighted_accuracy = batch_count = 0

        for batch in train_dataloader:
            batch_count += 1
            if isinstance(loss_fn, nn.MSELoss):
                y = batch[-1].to(device)
            else:
                y = batch[-1].long().to(device)

            if len(batch) > 2:
                x = tuple(t.to(device) for t in batch[:-1])
            else:
                x = batch[0].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze()

            if len(y_shape) == 1 or len(y_shape) == 3:
                loss = loss_fn(logits, y)
            elif len(y_shape) == 2:
                loss = loss_fn(logits.flatten(0, 1), y.flatten())

            batch_loss = loss.item()
            total_loss += batch_loss * y.shape[0]

            if isinstance(loss_fn, nn.MSELoss):
                abs_err = torch.sum(torch.abs(logits - y)).item()
                total_abs_err += abs_err
                batch_accuracy = abs_err / y.shape[0]
            elif accuracy_fn is None:
                if len(y_shape) == 1 or len(y_shape) == 3:
                    preds = torch.argmax(logits, dim=1)
                elif len(y_shape) == 2:
                    preds = torch.argmax(logits, dim=2)
                batch_accuracy = (preds == y).cpu().numpy().mean() * 100
                total_correct += (preds == y).sum().item()
            else:
                batch_accuracy = accuracy_fn(logits, y).item() * 100
                weighted_accuracy += batch_accuracy * len(y)

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if lr_history is not None:
                lr_history.append(optimizer.param_groups[0]['lr'])

            if batch_count % n_batch_per_report == 0:
                elapsed = time.time() - t0_batch
                print(f"{epoch:^7} | {batch_count:^7} | {batch_loss:^12.6f} | "
                      f"{'-':^10} | {batch_accuracy:^9.2f} | {'-':^9} | {elapsed:^9.2f}")
            t0_batch = time.time()

        if n_batch_per_report < batch_count:
            print("-" * 80)

        history["total_train_loss"].append(total_loss)
        history["average_train_loss"].append(total_loss / len(train_dataloader.dataset))

        if isinstance(loss_fn, nn.MSELoss):
            history["train_accuracy"].append(total_abs_err / len(train_dataloader.dataset))
        elif accuracy_fn is None:
            history["train_accuracy"].append(
                100 * total_correct / (len(train_dataloader.dataset) * n_labels_per_instance))
        else:
            history["train_accuracy"].append(weighted_accuracy / len(train_dataloader.dataset))

        if evaluation:
            model.eval()
            val_correct = val_loss = total_abs_err = weighted_accuracy = 0
            for batch in val_dataloader:
                if isinstance(loss_fn, nn.MSELoss):
                    y = batch[-1].to(device)
                else:
                    y = batch[-1].long().to(device)
                if len(batch) > 2:
                    x = tuple(t.to(device) for t in batch[:-1])
                else:
                    x = batch[0].to(device)
                with torch.no_grad():
                    logits = model(x).squeeze()
                if len(y_shape) == 1 or len(y_shape) == 3:
                    loss = loss_fn(logits, y)
                elif len(y_shape) == 2:
                    loss = loss_fn(logits.flatten(0, 1), y.flatten())
                val_loss += loss.item() * len(y)
                if isinstance(loss_fn, nn.MSELoss):
                    total_abs_err += torch.sum(torch.abs(logits - y)).item()
                elif accuracy_fn is None:
                    if len(y_shape) == 1 or len(y_shape) == 3:
                        preds = torch.argmax(logits, dim=1)
                    elif len(y_shape) == 2:
                        preds = torch.argmax(logits, dim=2)
                    val_correct += (preds == y).cpu().numpy().sum().item()
                else:
                    ba = accuracy_fn(logits, y).item()
                    weighted_accuracy += ba * len(y)

            history["total_val_loss"].append(val_loss)
            history["average_val_loss"].append(val_loss / len(val_dataloader.dataset))

            if isinstance(loss_fn, nn.MSELoss):
                history["val_accuracy"].append(total_abs_err / len(val_dataloader.dataset))
            elif accuracy_fn is None:
                history["val_accuracy"].append(
                    100 * val_correct / (len(val_dataloader.dataset) * n_labels_per_instance))
            else:
                history["val_accuracy"].append(
                    100 * weighted_accuracy / len(val_dataloader.dataset))

            elapsed = time.time() - t0_epoch
            print(f"{epoch:^7} | {'-':^7} | {history['average_train_loss'][-1]:^12.6f} | "
                  f"{history['average_val_loss'][-1]:^10.6f} | "
                  f"{history['train_accuracy'][-1]:^9.2f} | "
                  f"{history['val_accuracy'][-1]:^9.2f} | {elapsed:^9.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                if callbacks:
                    for cb in callbacks:
                        cb.on_best(model, optimizer)
        print("-" * 80)

    if best_state is not None:
        model.load_state_dict(best_state)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    return history
