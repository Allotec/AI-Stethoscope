#!/usr/bin/env python3

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.io.wavfile import read
from sklearn.preprocessing import OneHotEncoder

CPUS = mp.cpu_count()


def process_data(file_name):
    x_data = []
    y_data = []

    original_arr, sample_rate = librosa.load(file_name, sr=4000)
    original_arr = original_arr.reshape(1, len(original_arr))

    # Loop through segments of each file
    # Each segment will be 10000 elements long which is 2.5 seconds of audio recording
    for i in range(0, len(original_arr[0]) - 10000, 10000):
        arr = original_arr[0][i : i + 10000]

        # Generate MFCCs
        # Doing it this way makes the mfccs shape (20, 137)
        mfccs = librosa.feature.mfcc(y=arr, sr=sample_rate)

        # Generate spectrogram image
        spectrogram = plt.specgram(arr, Fs=sample_rate)[0]
        plt.close()

        mfccs_resized = (
            torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dimensions

        # Resize using interpolation to match the dimensions
        mfccs_resized = torch.nn.functional.interpolate(
            mfccs_resized,
            size=(spectrogram.shape[0], spectrogram.shape[1]),
            mode="nearest",
        )

        # Convert spectrogram and mfccs to PyTorch tensors
        spectrogram_tensor = (
            torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)
        )  # Add batch and channel dimensions

        # Concatenate along a new dimension to create a tensor with two channels
        combined_tensor = torch.cat((spectrogram_tensor, mfccs_resized), dim=1).squeeze(
            0
        )

        # bell = 0, diaphragm = 1, extended = 2
        mode = 0
        if file_name.startswith("D"):
            mode = 1
        elif file_name.startswith("E"):
            mode = 2

        diagnosis = file_name.split("_")[1].split(",")[0].lower()

        x_data.append([combined_tensor, mode])
        y_data.append(diagnosis)

    return x_data, y_data


def preprocess(directories):
    x_data = []
    y_data = []

    for directory in directories:
        files = [
            file for file in os.listdir(directory) if not ("and" in file or "+" in file)
        ]
        to_path = lambda file: os.path.join(directory, file)
        files = list(map(to_path, files))

        for file_path in files:
            x, y = process_data(file_path)
            x_data.extend(x)
            y_data.extend(y)

    return x_data, y_data


dataset = os.path.join("dataset", "Audio Files")
directories = [dataset]

print("Preprocessing Started")

x_data, y_data = preprocess(directories)

print("Preprocessing Done")

# Print count of y_data
unique, counts = np.unique(y_data, return_counts=True)
print(np.asarray((unique, counts)).T)

y_data = np.array(y_data).reshape(-1, 1)

# Perform one hot encoing on y_data
enc = OneHotEncoder()
y_data = enc.fit_transform(y_data).toarray()
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42
)

# Create datasest for train and test data
train_ds = CustomDataset(x_train, y_train)
test_ds = CustomDataset(x_test, y_test)

# Create DataLoaders for traing and test data
batch_size = 150
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

import torch.nn as nn


class DiagnosisNetwork(nn.Module):
    def __init__(self):
        super(DiagnosisNetwork, self).__init__()

        self.final_output_size = 8
        self.scalar_output_size = 32
        self.combined_input_size = 312

        self.convolution_pipeline = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=(5, 5), stride=(4, 2), padding=0),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.01),
            nn.Conv2d(24, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
        )

        self.scalar_pipeline = nn.Sequential(
            nn.Linear(1, self.scalar_output_size), nn.LeakyReLU(0.01)
        )

        self.combined_pipeline = nn.Sequential(
            nn.Linear(self.combined_input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(64, self.final_output_size),
            nn.Softmax(dim=1),
        )

        # Combine features from both inputs
        # self.combine_layer = nn.Linear(64 + 32, self.output_size)  # Adjust the output size as needed

    def forward(self, input_2d_array, input_scalar):

        # Process 2D array input with convolution
        conv_flat = self.convolution_pipeline(input_2d_array)
        # final_output = self.no_mode_post_convolution(conv_flat)

        # Process scalar input
        scalar_output = self.scalar_pipeline(input_scalar)

        # Concatenate the outputs of both branches
        combined_output = torch.cat((conv_flat, scalar_output), dim=1)

        # Final output layer
        final_output = self.combined_pipeline(combined_output)

        return final_output


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiagnosisNetwork().to(device)

lossfn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

loss_vals = []
EPOCHS = 1000

from tqdm import tqdm

for epoch in range(EPOCHS):
    epoch_loss = 0

    with tqdm(train_dl, unit="batch") as tepoch:
        for [x, y] in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            input_2d_array, input_scalar = x

            # Reshape mode size for network
            input_scalar = input_scalar.unsqueeze(1)

            # Correct types and send to device
            input_scalar = input_scalar.float()
            input_2d_array, input_scalar, y = (
                input_2d_array.to(device),
                input_scalar.to(device),
                y.to(device),
            )

            # forward pass
            pred_probab = model(input_2d_array, input_scalar)

            # calculate loss
            loss = lossfn(pred_probab, y)

            # backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            tepoch.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / len(train_dl)
        loss_vals.append(epoch_avg_loss)

correct = 0
total = 0

with torch.no_grad():
    model.eval()
    for batch in test_dl:
        # grab data
        x, y = batch
        input_2d_array, input_scalar = x

        # Reshape image and mode size for network
        input_scalar = input_scalar.unsqueeze(1)

        # Correct types and send to device
        input_scalar = input_scalar.float()
        input_2d_array, input_scalar, y = (
            input_2d_array.to(device),
            input_scalar.to(device),
            y.to(device),
        )
        input_2d_array, y = input_2d_array.to(device), y.to(device)

        pred_probab = model(input_2d_array, input_scalar)
        yhat = pred_probab.argmax(1).float()
        total += y.size(0)
        correct += (yhat == y.argmax(1)).sum().item()


accuracy = 100 * (correct / total)

print(f"accuracy: {accuracy}%")

epochs = [x for x in range(len(loss_vals))]
window_size = int(EPOCHS / 100)
moving_average = np.convolve(
    loss_vals, np.ones(window_size) / window_size, mode="valid"
)

plt.plot(
    epochs[window_size - 1 :],
    moving_average,
    linestyle="-",
    color="b",
    label="Loss",
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss Over Epochs")
plt.xticks(epochs[:: int(len(epochs) / 10)])
plt.grid()
plt.legend()
plt.savefig("loss.png", dpi=300)
