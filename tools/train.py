import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from models.phase_classifier import PhaseClassifier
from datasets.ising_dataset import IsingDataset


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train_1epoch(model, dataloader, optimizer, criterion):
    model.train()
    batch_loss = []
    with torch.set_grad_enabled(True):
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

    return sum(batch_loss) / len(batch_loss)


def validate_1epoch(model, dataloader, criterion):
    model.eval()
    batch_loss = []
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            batch_loss.append(loss.item())

    return sum(batch_loss) / len(batch_loss)


if __name__ == '__main__':
    seed = 42
    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters
    epoch_num = 50
    Nx = 20
    Ny = 20
    Ts = [T/10 for T in range(10, 41, 5)]

    # create datasets
    train_dataset = IsingDataset(Nx=Nx, Ny=Nx, Ts=Ts, phase='train', data_num=512)
    val_dataset = IsingDataset(Nx=Nx, Ny=Nx, Ts=Ts, phase='val', data_num=128)

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # create model
    model = PhaseClassifier(Nx*Ny, 100)
    model.to(device)

    # create loss function
    criterion = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # loop run epoch
    train_loss = []
    val_loss = []
    for i in range(1, epoch_num+1):
        # train 1epoch
        loss = train_1epoch(model, train_loader, optimizer, criterion)
        train_loss.append(loss)

        # validate 1epoch
        loss = validate_1epoch(model, val_loader, criterion)
        val_loss.append(loss)

        # print progress
        print(f'epoch : {i:>2}/{epoch_num}, \
                train loss : {train_loss[-1]}, \
                val loss : {val_loss[-1]}')

    # save model weight
    torch.save(model.state_dict(), "model.pth")

    # save learning curve
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.plot()
    plt.savefig('loss.png')

    # clear figure
    plt.clf()

    # detect critical temperature
    Plow = []       # probabilities higher than critical temperature
    Phigh = []      # probabilities lower than critical temperature
    m = nn.Softmax(dim=1)
    for T in Ts:
        dataset = IsingDataset(Nx=20, Ny=20, Ts=[T], phase='val', data_num=128)
        data, _ = dataset[:]
        preds = m(model(data))
        preds = preds.cpu().detach().numpy()
        Plow.append(np.mean(preds[:,0]))
        Phigh.append(np.mean(preds[:,1]))

    plt.title('2D square lattice Ising model')
    plt.xlabel("Tempreture")
    plt.ylabel("Probability")
    plt.plot(Ts, Plow,  linestyle='--', marker='o', color='b', label='low')
    plt.plot(Ts, Phigh, linestyle='--', marker='o', color='r', label='high')
    plt.legend()
    plt.plot()
    plt.savefig('2D_square_lattice_Ising_model.png')