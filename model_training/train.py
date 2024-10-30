#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from neural_network import *
from torchsummary import summary
from torchview import draw_graph

TEST_AMOUNT = 0.2
BATCH_SIZE = 30
DIV_FAC = 10
EPOCHS = 1000
LOSS_SMOOTHING = EPOCHS // DIV_FAC

IMG_DIR = ".out_files"


def main():
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    directories = [os.path.join("dataset", "Audio Files")]
    x_data, y_data = preprocess(directories)

    print("Loading Audio Data")
    train_dl, test_dl = get_datasets(
        x_data, y_data, TEST_AMOUNT, BATCH_SIZE, CustomDataset
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiagnosisNetwork().to(device)
    lossfn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    # TODO: Get the summary fully working
    # print_summary(model)
    output_graph_png(model, train_dl)

    loss_vals = []
    accuracy_hist = []

    top_1 = evaluate(device, test_dl, model)
    accuracy_hist.append(top_1)
    for i in range(1):
        loss = train(
            device,
            train_dl,
            optim,
            lossfn,
            model,
            LOSS_SMOOTHING,
            LOSS_SMOOTHING * i,
        )
        top_1 = evaluate(device, test_dl, model)

        loss_vals.extend(loss)
        accuracy_hist.append(top_1)

    accuracy_img = os.path.join(IMG_DIR, "accuracy.png")
    plot_accuracy(accuracy_hist, accuracy_img)

    loss_img = os.path.join(IMG_DIR, "loss.png")
    plot_loss(loss_vals, LOSS_SMOOTHING, loss_img)


def print_summary(model):
    input_2d_shape = (2, 129, 77)
    input_scalar_shape = (1,)
    size = np.array([input_2d_shape, input_scalar_shape], dtype="object")
    summary(model, input_size=size, batch_size=BATCH_SIZE)


def plot_accuracy(accuracy, output_path):
    epochs = [x for x in range(len(accuracy))]

    plt.plot(
        epochs,
        accuracy,
        linestyle="-",
        color="b",
        label="Accuracy",
    )

    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Top 1 Accuracy")

    epochs = epochs.append(EPOCHS)
    plt.xticks(epochs)
    plt.ylim(0, 100)

    plt.grid()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


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


def get_data_sample(train_dl):
    [x, y] = next(iter(train_dl))
    input_2d_array, input_scalar = x
    input_scalar = input_scalar.unsqueeze(1).float()

    return input_2d_array, input_scalar


def output_graph_png(model, train_dl):
    input_data, scalar = get_data_sample(train_dl)
    draw_graph(
        model,
        input_data=input_data,
        input_scalar=scalar,
        graph_dir="LR",
        save_graph=True,
        directory=IMG_DIR,
    )


if __name__ == "__main__":
    main()
