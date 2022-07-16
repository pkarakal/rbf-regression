import sys
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from rbf_boston_housing.utils import get_boston, split_dataset, train, evaluate, draw_loss
from torch import nn
from rbf_boston_housing.nn import RBFNetwork
from rbf_boston_housing.cli_parser import Parser

if __name__ == "__main__":
    parser = Parser()
    cli = parser.parseCLI(sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    pin_memory = False

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    print(f"Will be using {device} for training and testing")

    batch_size = int(cli.get('batch_size'))
    n_epochs = int(cli.get('epochs'))
    X_train, X_test, Y_train, Y_test = split_dataset(*get_boston())
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    train_set = TensorDataset(torch.tensor(X_train.to_numpy(), dtype=torch.float), torch.tensor(Y_train.to_numpy(), dtype=torch.float).view(-1, 1))
    test_set = TensorDataset(torch.tensor(X_test.to_numpy(), dtype=torch.float), torch.tensor(Y_test.to_numpy(), dtype=torch.float).view(-1, 1))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=pin_memory)
    dropout = float(cli.get("dropout"))
    boston_model = RBFNetwork(X_train.to_numpy().shape[1], dropout if dropout is not None else 0.2)
    criterion = nn.SoftMarginLoss()
    rate = float(cli.get('rate'))
    optimizer = torch.optim.SGD(boston_model.parameters(), lr=rate if rate is not None else 0.002)
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)
    boston_model.to(device)
    train_loss = train(model=boston_model, device=device, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=n_epochs)
    test_loss = evaluate(model=boston_model, device=device, test_loader=test_loader, criterion=criterion)
    draw_loss(train_loss, test_loss)
    if cli.get('save_model'):
        torch.save(boston_model.state_dict(), "boston.pt")
