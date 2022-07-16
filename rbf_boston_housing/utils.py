import time

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch import max, no_grad
import matplotlib.pyplot as plt


def get_boston() -> (pd.DataFrame, pd.DataFrame):
    boston = load_boston()
    data = pd.DataFrame(boston['data'])
    data.columns = boston['feature_names']
    return pd.DataFrame(boston['target']), data.iloc[:, 0:13]


def split_dataset(prices: pd.DataFrame, features: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    return train_test_split(features, prices, test_size=0.3, random_state=42)


def __train__(model, device, train_loader, criterion, optimizer, epochs=100) -> list:
    loss_val = []
    model.train()  # prep model for training

    for epoch in range(epochs):
        # monitor training loss
        train_loss = 0.0
        for features, prices in train_loader:
            features, prices = features.to(device), prices.to(device)
            features = features.view(features.shape[0], -1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(features)
            # calculate the loss
            loss = criterion(output, prices)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            val = loss.item() * features.size(0)
            loss_val.append(val)
            train_loss += val

        train_loss = train_loss / len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch + 1,
            train_loss
        ))
    return loss_val


def __evaluate__(model, device, test_loader, criterion) -> list:
    test_loss = 0.0
    y_pred = []
    y_true = []
    eval_loss = []
    model.eval()  # prep model for *evaluation*

    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(target.cpu().numpy())
            loss = criterion(output, target)
            el = loss.item() * data.size(0)
            eval_loss.append(el)
            test_loss += el
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    test_loss = test_loss / len(test_loader.dataset)
    print('Average Test Loss: {:.6f}\n'.format(test_loss))
    print('R2 score: {:.6f}'.format(r2))
    print('MSE score: {:.6f}'.format(mse))
    print('MAE score: {:.6f}'.format(mae))
    return eval_loss


def train(model, device, train_loader, criterion, optimizer, epochs=100) -> list:
    start = time.time()
    train_loss = __train__(model, device, train_loader, criterion, optimizer, epochs)
    end = time.time()
    print(f"Training took {end-start}s")
    return train_loss


def evaluate(model, device, test_loader, criterion) -> list:
    start = time.time()
    test_loss = __evaluate__(model, device, test_loader, criterion)
    end = time.time()
    print(f"Evaluation took {end-start}s")
    return test_loss


def stopwatch(func):
    start = time.time()
    func()
    end = time.time()
    return f"{end - start}"


def draw_loss(train_loss, test_loss):
    plt.figure(1)
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.title('Learning curves of training and validation sets, 1st network version')
    plt.show()
