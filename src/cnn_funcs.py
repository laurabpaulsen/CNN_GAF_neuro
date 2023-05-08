from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import random

import matplotlib.pyplot as plt

# sklearn tools
from sklearn.model_selection import train_test_split

# pytorch tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# set parameters for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300


def load_gaf(file: Path):
    """
    Load a gaf image from path and return it as a numpy array

    Parameters
    ----------
    file : Path
        Path to gaf
    
    Returns
    -------
    gaf : np.array
        The gaf image
    label : int
        Label indicating the label (animate or inanimate)
    """
    
    gaf = np.load(file)
    label = str(file)[-5]
    
    return gaf, int(label)


def load_gafs(gaf_path: Path, n_jobs: int = 1, all_subjects=False):
    """
    Loads gaf images from path and return them as a numpy array using multiprocessing

    Parameters
    ----------
    gaf_path : Path
        Path to gaf images
    n_jobs : int, optional
        Number of jobs to use for multiprocessing, by default 1
    
    Returns
    -------
    gafs : np.array
        The gaf images
    labels : np.array
        Labels indicating the label (animate or inanimate) for each gaf image
    """
    gafs = []
    labels = []

    files = list(gaf_path.iterdir())

    if all_subjects:
        files = [list(dir.iterdir()) for dir in files]
        files = [item for sublist in files for item in sublist]
        files = random.choices(files, k=10000) # choosing 50000 random trials to avoid memory overload

    
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            for gaf, label in tqdm(pool.imap(load_gaf, files), total=len(files), desc="Loading in data"):
                gafs.append(gaf)
                labels.append(label)
    else:
        for file in tqdm(files, desc="Loading in data"):
            gaf, label = load_gaf(file)
            gafs.append(gaf)
            labels.append(label)
    
    return np.array(gafs), np.array(labels)


class GAFDataset(Dataset):
    """Dataset class for GAF images, inherits from torch.utils.data.Dataset"""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y
 
def prep_dataloaders(gafs, labels, batch_size=4):
    """
    Creates dataloaders for training, validation, and testing

    Parameters
    ----------
    gafs : np.array
        The gaf images
    labels : np.array
        array of labels
    batch_size : int, optional
        Batch size, by default 4

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Testing data loader
    y_test : np.array
        Labels for the test set
    """

    # split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(gafs, labels, test_size=0.2, random_state=7)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=7)

    # create dataloaders
    train_loader = DataLoader(GAFDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GAFDataset(X_val, y_val), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GAFDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_test


# define a class for the CNN
class CNN():
    # initialize the class
    def __init__(self, model, optimizer, criterion, lr = 0.001):
        self.lr = lr
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    # train the model for one epoch
    def train(self, train_loader:DataLoader):
        """
        Train the model for one epoch and return the loss and accuracy

        Parameters
        ----------
        train_loader : DataLoader
            The training data loader
        
        Returns
        -------
        train_loss : float
            The training loss
        train_acc : float
            The training accuracy
        """
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for X, y in train_loader:
            self.optimizer.zero_grad()
            y_hat = self.model(X.float())
            loss = self.criterion(y_hat.view(-1), y.float())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += ((torch.round(torch.sigmoid(y_hat.view(-1)))==y).sum().item() / len(y))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        return train_loss, train_acc
    
    # train the model for X epochs
    def train_model(self, train_loader:DataLoader, val_loader:DataLoader, epochs: int):
        """Train the model and return the losses and accuracies
        
        Parameters
        ----------
        train_loader : DataLoader
            The training data loader
        val_loader : DataLoader
            The validation data loader
        epochs : int
            The number of epochs to train for
        
        Returns
        -------
        history : dict
            Dictionary with the train and validation loss and accuracies. 
        """

        # dict for storing losses and accuracies
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            # train
            train_loss, train_acc = self.train(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # validate
            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

        return history
    
    def validate(self, val_loader:DataLoader): 
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for X, y in val_loader:
                y_hat = self.model(X.float())
                loss = self.criterion(y_hat.view(-1), y.float())
                val_loss += loss.item()
                val_acc += ((torch.round(torch.sigmoid(y_hat.view(-1)))==y).sum().item() / len(y))

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

        return val_loss, val_acc
    

    def predict(self, test_loader):
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
                y_hat = self.model(X.float())
                y_pred.append(torch.sigmoid(y_hat).numpy())

        return np.concatenate(y_pred)
    
    def state_dict(self):
        return self.model.state_dict()


def prep_model(lr: float):
    """
    Initialize CNN, optimizer, and loss function

    Parameters
    ----------
    lr : float
        Learning rate
    """
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv3d(50, 18, kernel_size=3)
            self.pool1 = nn.MaxPool3d(kernel_size=1)
            self.bn1 = nn.BatchNorm3d(18)
            self.drop1 = nn.Dropout3d(p=0.2)

            self.conv2 = nn.Conv3d(18, 128, kernel_size=(3, 3, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn2 = nn.BatchNorm3d(128)
            self.drop2 = nn.Dropout3d(p=0.2)
            
            self.conv3 = nn.Conv3d(128, 128, kernel_size=(3, 3, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn3 = nn.BatchNorm3d(128)
            self.drop3 = nn.Dropout3d(p=0.2)
            

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
            x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv2(x)))))
            x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv3(x)))))
            x = self.avgpool(x)
            x = x.view(-1, 128)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Net()

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion





def plot_history(history, save_path = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(history['train_loss'], label="Train Loss")
    ax[0].plot(history['val_loss'], label="Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history['train_acc'], label="Train Accuracy")
    ax[1].plot(history['val_acc'], label="Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

