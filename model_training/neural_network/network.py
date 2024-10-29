import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
