import os

import librosa
import matplotlib
import matplotlib.pyplot as plt
import torch

BELL = 0
DIAPHRAGM = 1
EXTENDED = 2


def data_from_file(file_name):
    matplotlib.use("agg")

    x_data = []

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

        # NOTE: I dont understand the different options for mode
        x_data.append([combined_tensor, DIAPHRAGM])

    return x_data


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

        mode = BELL
        if file_name.startswith("D"):
            mode = DIAPHRAGM
        elif file_name.startswith("E"):
            mode = EXTENDED

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
