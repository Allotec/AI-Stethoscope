import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_datasets(x_data, y_data, test_amount, batch_size, data_set_class):
    y_data = np.array(y_data).reshape(-1, 1)

    # Perform one hot encoing on y_data
    y_data = OneHotEncoder().fit_transform(y_data).toarray()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_amount, random_state=42
    )

    # Create datasest for train and test data
    train_ds = data_set_class(x_train, y_train)
    test_ds = data_set_class(x_test, y_test)

    # Create DataLoaders for traing and test data
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl


def train(device, train_dl, optim, lossfn, model, epochs, epoch_start):
    loss_vals = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        with tqdm(train_dl, unit="batch") as tepoch:
            for [x, y] in tepoch:
                current_epoch = epoch_start + epoch
                tepoch.set_description(f"Epoch {current_epoch}")
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

    return loss_vals


def evaluate(device, test_dl, model):
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()

        for batch in test_dl:
            # grab data
            x, y = batch
            input_2d_array, input_scalar = x

            # Reshape scalar input for network and ensure correct types
            input_scalar = input_scalar.unsqueeze(1).float()

            # Send tensors to the device
            input_2d_array, input_scalar, y = (
                input_2d_array.to(device),
                input_scalar.to(device),
                y.to(device),
            )

            # Get model predictions
            pred_probab = model(input_2d_array, input_scalar)
            yhat = pred_probab.argmax(1).float()

            # Count correct predictions
            total += y.size(0)
            correct += (yhat == y.argmax(1)).sum().item()

    top1 = 100 * (correct / total)
    return top1
