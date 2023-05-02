# data tools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# pytorch tools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class GAFDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

def load_gafs(gaf_path):
    """Load gaf images from path and return them as a numpy array"""
    gafs = []
    labels = []

    files = list(gaf_path.iterdir())

    for file in tqdm(files[:10000], desc="Loading in data"):
        if file.is_file():
            gaf = np.load(file)
            gafs.append(gaf)

            label = str(file)[-5]
            labels.append(int(label))
    gafs = np.array(gafs)
    labels = np.array(labels)
    
    return gafs, labels

def prep_model(height = 50, width = 50, depth=63):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv3d(3, 18, kernel_size=6)
            self.pool1 = nn.MaxPool3d(kernel_size=2)
            self.bn1 = nn.BatchNorm3d(18)
            self.drop1 = nn.Dropout3d(p=0.2)

            self.conv2 = nn.Conv3d(18, 128, kernel_size=6)
            self.pool2 = nn.MaxPool3d(kernel_size=2)
            self.bn2 = nn.BatchNorm3d(128)
            self.drop2 = nn.Dropout3d(p=0.2)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
            x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv2(x)))))
            x = self.avgpool(x)
            x = x.view(-1, 128)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):
    # train the model
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(X.float())
            loss = criterion(y_hat.view(-1), y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += ((torch.round(torch.sigmoid(y_hat)) == y).sum().item() / len(y))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for X, y in val_loader:
                y_hat = model(X.float())
                loss = criterion(y_hat.view(-1), y.float())
                val_loss += loss.item()
                val_acc += ((torch.round(torch.sigmoid(y_hat)) == y).sum().item() / len(y))

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def balance_classes(X, y):
    """
    Balances the class weight by removing trials from classes with more trials

    Parameters
    ----------
    X : array
        Data array with n_trials as the first dimension
    y : array
        Array with shape (n_trials, )
    
    Returns
    -------
    X_equal : array
        Data array with a equal number of trials for each class
    y_equal : array
        Array with shape (n_trials, ) containing classes with equal number of trials for each class
    """
    keys, counts = np.unique(y, return_counts = True)

    # get the minimum number of trials
    min_count = np.min(counts)

    # loop through each class and remove trials
    remove_ind = []
    for key, count in zip(keys, counts):
        index = np.where(np.array(y) == key)
        random_choices = np.random.choice(len(index[0]), size = count-min_count, replace=False)
        remove_ind.extend([index[0][i] for i in random_choices])
    
    X_equal = np.delete(X, remove_ind, axis = 0)
    y_equal = np.delete(y, remove_ind, axis = 0)

    return X_equal, y_equal
