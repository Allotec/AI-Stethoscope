#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from neural_network import *

TEST_AMOUNT = 0.2
BATCH_SIZE = 150
EPOCHS = 1000
LOSS_SMOOTHING = 100

IMG_DIR = ".out_files"


def main():
    directories = [os.path.join("dataset", "Audio Files")]
    x_data, y_data = preprocess(directories)

    train_dl, test_dl = get_datasets(
        x_data, y_data, TEST_AMOUNT, BATCH_SIZE, CustomDataset
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiagnosisNetwork().to(device)
    lossfn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    loss_vals = train(device, train_dl, optim, lossfn, model, EPOCHS)
    accuracy = test_accuracy(device, test_dl, model)
    print(f"Top 1 Accuracy: {accuracy}%")

    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    loss_img = os.path.join(IMG_DIR, "loss.png")
    plot_loss(loss_vals, LOSS_SMOOTHING, loss_img)


def plot_loss(loss_vals, smoothing, output_path):
    epochs = [x for x in range(len(loss_vals))]
    epochs = list(map(lambda x: x + 1, epochs))

    moving_average = np.convolve(
        loss_vals, np.ones(smoothing) / smoothing, mode="valid"
    )

    plt.plot(
        epochs[smoothing - 1 :],
        moving_average,
        linestyle="-",
        color="b",
        label="Loss",
    )

    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Smoothed")

    plt.xticks(epochs[:: int(len(epochs) / 10)].append(EPOCHS))
    plt.ylim(0, max(loss_vals))

    plt.grid()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
