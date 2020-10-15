from math import floor

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import display
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
plt.rcParams.update({"font.size": 14})
pd.set_option("display.max_rows", None, "display.max_columns", None)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def summary(self):
        names = [name for name, _ in self.named_parameters()] + ["Total"]
        params = [param.numel() for _, param in self.named_parameters()]
        params += [sum(params)]
        params_df = pd.DataFrame(list(zip(names, params)), columns=["Layer", "Parameters"])
        print(params_df)


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

    if not isinstance(d, dict):
        return d

    class C:
        pass

    obj = C()
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj


def split_data(x_train, y_train, train_size, val_size):
    return x_train[:train_size], y_train[:train_size], x_train[-val_size:], y_train[-val_size:]


def prepare_data(train_file, test_file, batch_size, train_size, val_size, test_size):
    # Read data from files
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Split x and y for train and test set
    x_train, y_train = train_data.iloc[:, 1:].to_numpy(), train_data["label"].to_numpy()
    x_test, y_test = test_data.iloc[:, 1:].to_numpy(), test_data["label"].to_numpy()

    # Shuffle traing and test datasets
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # Split train, validation, and test set
    x_train, y_train, x_val, y_val = split_data(x_train, y_train, train_size, val_size)
    x_test, y_test, _, _ = split_data(x_test, y_test, test_size, 1)

    # Normalize
    x_train = x_train.astype("float32") / 255
    x_val = x_val.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert to tensors
    x_train, y_train, x_val, y_val, x_test, y_test = map(
        torch.from_numpy, [x_train, y_train, x_val, y_val, x_test, y_test]
    )

    assert len(x_train) == len(y_train)
    assert len(x_val) == len(y_val)
    assert len(x_test) == len(y_test)

    print(f"Train size: {len(x_train)}")
    print(f"Val size: {len(x_val)}")
    print(f"Test size: {len(x_test)}")

    # Create tensor datasets
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, pin_memory=True)

    return (train_loader, val_loader, test_loader)


def accuracy(pred, y):
    pred_class = pred.argmax(1, keepdim=True)
    correct = pred_class.eq(y.view_as(pred_class)).sum()
    acc = correct.float() / y.size(0)
    return acc


def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in data_loader:
        X, y = batch
        X, y = X.to(device), y.to(device)
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        pred = model(X)
        loss = criterion(pred, y)
        acc = accuracy(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    avg_loss = epoch_loss / len(data_loader)
    avg_acc = epoch_acc / len(data_loader)
    return (avg_loss, avg_acc)


def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            acc = accuracy(pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    avg_loss = epoch_loss / len(data_loader)
    avg_acc = epoch_acc / len(data_loader)
    return (avg_loss, avg_acc)


def run(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = [
        config.train_file,
        config.test_file,
        config.batch_size,
        config.train_size,
        config.val_size,
        config.test_size,
    ]
    train_loader, val_loader, test_loader = prepare_data(*data_config)

    # Define model, optimizer, and loss function
    model = Model(config).to(device)
    model.summary()
    print()
    print(model)
    print()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_log = []
    val_loss_log = []
    train_acc_log = []
    val_acc_log = []

    train_log = pd.DataFrame(
        columns=["Epoch", "Train loss", "Train acc", "Validation loss", "Validation acc"]
    )

    # Train
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_log.loc[epoch] = [epoch + 1] + [train_loss, train_acc, val_loss, val_acc]

        # HTML(train_log.to_html(index=False))
        display(train_log)
        print()

        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        train_acc_log.append(train_acc)
        val_acc_log.append(val_acc)

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:>5.4f}  Test Acc: {test_acc * 100:>5.4f}%")

    return (train_loss_log, val_loss_log, train_acc_log, val_acc_log)


config = {
    "hidden_dim": 256,
    "train_file": "mnist_train.csv",
    "test_file": "mnist_train.csv",
    "batch_size": 128,
    "train_size": 55000,
    "val_size": 5000,
    "test_size": 10000,
    "lr": 1e-3,
    "epochs": 30,
}

config = dict2obj(config)

(train_loss_log, val_loss_log, train_acc_log, val_acc_log) = run(config)
