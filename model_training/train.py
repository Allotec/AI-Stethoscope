#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from neural_network import *

TEST_AMOUNT = 0.2
BATCH_SIZE = 30
DIV_FAC = 10
EPOCHS = 1000
STEPS = EPOCHS // DIV_FAC

IMG_DIR = ".out_files"


def main():
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    print("Preprocessing Audio data")
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

    model_summary(model, optim)

    loss_vals = []
    accuracy_hist = []

    top_1 = evaluate(device, test_dl, model)
    accuracy_hist.append(top_1)
    for i in range(DIV_FAC):
        loss = train(
            device,
            train_dl,
            optim,
            lossfn,
            model,
            STEPS,
            STEPS * i,
        )
        top_1 = evaluate(device, test_dl, model)

        loss_vals.extend(loss)
        accuracy_hist.append(top_1)

    accuracy_img = os.path.join(IMG_DIR, "accuracy.png")
    plot_accuracy(accuracy_hist, accuracy_img)

    loss_img = os.path.join(IMG_DIR, "loss.png")
    plot_loss(loss_vals, loss_img)

    weights_path = os.path.join(os.getcwd(), "model_weights.pt")
    torch.save(model.state_dict(), weights_path)


def model_summary(model, optim):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optim.state_dict():
        print(var_name, "\t", optim.state_dict()[var_name])


def plot_accuracy(accuracy, output_path):
    epochs = list(map(lambda x: x * STEPS, [x for x in range(len(accuracy))]))

    plt.plot(
        epochs,
        accuracy,
        linestyle="-",
        color="b",
        label="Accuracy",
    )

    plt.title("Model Top 1 Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Top 1 Accuracy")

    plt.ylim(0, 100)

    plt.grid()
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_loss(loss_vals, output_path):
    epochs = [x for x in range(len(loss_vals))]

    plt.plot(
        epochs,
        loss_vals,
        linestyle="-",
        color="b",
        label="Loss",
    )

    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Smoothed")

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


if __name__ == "__main__":
    main()
