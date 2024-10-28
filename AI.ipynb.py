from scipy.io.wavfile import read
import torchaudio
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import librosa

def preprocess(directories):
  x_data = []
  y_data = []
  for directory in directories:
    for filename in os.listdir(directory):
      f = os.path.join(directory, filename)

      if "and" in filename or "+" in filename:
        continue

      # Create 1d array from audio input
      original_arr, sample_rate = torchaudio.load(f)
      """
      print(sample_rate)
      print("here")
      print("filename: " + filename)
      print("Lenth original: " + str(len(original_arr[0])))
      print(original_arr)
      """
      # Loop through segments of each file
      # Each segment will be 10000 elements long which is 2.5 seconds of audio recording
      for i in range(0, len(original_arr[0])-10000, 10000):
        arr = original_arr[0][i:i+10000]

        arr = arr.view(-1)

        # Generate MFCCs
        # Doing it this way makes the mfccs shape (20, 137)
        mfccs = librosa.feature.mfcc(y=arr.numpy(), sr=sample_rate)

        # Generate spectrogram image
        spectrogram = plt.specgram(arr, Fs= sample_rate)[0]
        plt.close()

        mfccs_resized = torch.FloatTensor(mfccs).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Resize using interpolation to match the dimensions
        mfccs_resized = torch.nn.functional.interpolate(mfccs_resized, size=(spectrogram.shape[0], spectrogram.shape[1]), mode='nearest')


        # Plotting original mfccs vs resized mfccs - commented out cause it takes forever just uncomment it if you wanna see the images
        """
        plt.figure(figsize=(10, 5))

        # Plot for the first graph
        plt.subplot(1, 2, 1)
        plt.plot(mfccs)
        plt.title('Original')
        plt.xlabel('X')
        plt.ylabel('Y')

        # Plot for the second graph
        plt.subplot(1, 2, 2)
        plt.plot(mfccs_resized.squeeze().numpy())
        plt.title('Resized')
        #plt.colorbar(label='Value')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.show()
        """

        # Convert spectrogram and mfccs to PyTorch tensors
        spectrogram_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Concatenate along a new dimension to create a tensor with two channels
        combined_tensor = torch.cat((spectrogram_tensor, mfccs_resized), dim=1).squeeze(0)


        # bell = 0, diaphragm = 1, extended = 2
        mode = 0
        if filename.startswith("D"):
          mode = 1
        elif filename.startswith("E"):
          mode = 2
        if directory == "/content/drive/MyDrive/ICBHI_final_database":
          mode = 1
          diagnosis = patient_diagnosis[int(filename.split("_")[0])]
        else:
          diagnosis = filename.split("_")[1].split(",")[0].lower()

        # Add x data row to x_data
        #image = torch.FloatTensor(image)
        x_data.append([combined_tensor, mode])

        # Get y and add it to y_data
        y_data.append(diagnosis)
  return x_data, y_data


#directories = ["/content/drive/MyDrive/training audio files", "/content/drive/MyDrive/ICBHI_final_database"]
directories = ["dataset\Audio Files"]
print("Preprocessing Started")
x_data, y_data = preprocess(directories)
print("Preprocessing Done")
print(y_data)
# Print count of y_data
unique, counts = np.unique(y_data, return_counts=True)
print(np.asarray((unique, counts)).T)
# trim some data away to balance it out
a = 0
copd = 0
hf = 0
n = 0
pn = 0
y_data_copy = []
x_data_copy = []
for i in range(len(y_data)):
  if y_data[i] == "asthma":
    if a < 100:
      a += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "copd":
    if copd < 100:
      copd += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "heart failure":
    if hf < 100:
      hf += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "n":
    if n < 100:
      n += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "pneumonia":
    if pn < 100:
      pn += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  else:
    y_data_copy.append(y_data[i])
    x_data_copy.append(x_data[i])

b = 42
lf = 69
pf = 57

unique, counts = np.unique(y_data_copy, return_counts=True)
print(np.asarray((unique, counts)).T)
# Sample diagnosis with less than 100 records multiple times
for i in range(len(y_data)):
  if y_data[i] == "bron":
    if b < 100:
      b += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "lung fibrosis":
    if lf < 100:
      lf += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])
  elif y_data[i] == "plueral effusion":
    if pf < 100:
      pf += 1
      y_data_copy.append(y_data[i])
      x_data_copy.append(x_data[i])

unique, counts = np.unique(y_data_copy, return_counts=True)
print(np.asarray((unique, counts)).T)
# Reshape y_data for encoding
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
from torch.utils.data import TensorDataset, DataLoader

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

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
        nn.Linear(1, self.scalar_output_size),
        nn.LeakyReLU(0.01)
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
        nn.Softmax(dim=1)
    )


    # Combine features from both inputs
    #self.combine_layer = nn.Linear(64 + 32, self.output_size)  # Adjust the output size as needed

  def forward(self, input_2d_array, input_scalar):

    # Process 2D array input with convolution
    conv_flat = self.convolution_pipeline(input_2d_array)
    #final_output = self.no_mode_post_convolution(conv_flat)

    # Process scalar input
    scalar_output = self.scalar_pipeline(input_scalar)

    # Concatenate the outputs of both branches
    combined_output = torch.cat((conv_flat, scalar_output), dim=1)

    # Final output layer
    final_output = self.combined_pipeline(combined_output)

    return final_output
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiagnosisNetwork().to(device)
print(device)

lossfn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
#optim = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)

for epoch in range(2000):
  for batch in train_dl:
    # grab data
    X, y = batch
    input_2d_array, input_scalar = X

    # Reshape mode size for network
    input_scalar = input_scalar.unsqueeze(1)

    # Correct types and send to device
    input_scalar = input_scalar.float()
    input_2d_array, input_scalar, y = input_2d_array.to(device), input_scalar.to(device), y.to(device)
    #input_2d_array, y = input_2d_array.to(device), y.to(device)

    # forward pass
    pred_probab = model(input_2d_array, input_scalar)
    #pred_probab = model(input_2d_array)

    # calculate loss
    loss = lossfn(pred_probab, y)

    # backpropagation
    optim.zero_grad()
    loss.backward()
    optim.step()

  # print loss for each epoch
  print(f"Epoch: {epoch}, Loss: {loss.item()}")
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for batch in test_dl:
        # grab data
        X, y = batch
        input_2d_array, input_scalar = X

        # Reshape image and mode size for network
        input_scalar = input_scalar.unsqueeze(1)

        # Correct types and send to device
        input_scalar = input_scalar.float()
        input_2d_array, input_scalar, y = input_2d_array.to(device), input_scalar.to(device), y.to(device)
        input_2d_array, y = input_2d_array.to(device), y.to(device)

        pred_probab = model(input_2d_array, input_scalar)
        yhat = pred_probab.argmax(1).float()
        total += y.size(0)
        correct += (yhat == y.argmax(1)).sum().item()
print(correct)
print(total)
accuracy = 100 * (correct/total)
print(f"accuracy: {accuracy}%")
