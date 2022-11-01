import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from models.phase_classifier import PhaseClassifier
from datasets.ising_dataset import IsingDataset


if __name__ == '__main__':
    # set temperatures
    Nx = 20
    Ny = 20
    Ts = [T/10 for T in range(25, 51, 5)]

    # set model
    model_path = 'model.pth'
    model = PhaseClassifier(Nx*Ny, 100)
    model.load_state_dict(torch.load(model_path))

    # detect critical temperature
    Plow = []       # probabilities higher than critical temperature
    Phigh = []      # probabilities lower than critical temperature
    m = nn.Softmax(dim=1)
    for T in Ts:
        dataset = IsingDataset(Nx=Nx, Ny=Ny, Ts=[T], phase='test', data_num=128)
        data, _ = dataset[:]
        preds = m(model(data))
        preds = preds.cpu().detach().numpy()
        Plow.append(np.mean(preds[:,0]))
        Phigh.append(np.mean(preds[:,1]))

    plt.title('2D triangular lattice Ising model')
    plt.xlabel("Temperature")
    plt.ylabel("Probability")
    plt.plot(Ts, Plow,  linestyle='--', marker='o', color='b', label='low')
    plt.plot(Ts, Phigh, linestyle='--', marker='o', color='r', label='high')
    plt.legend()
    plt.plot()
    plt.savefig('2D_triangular_lattice_Ising_model.png')