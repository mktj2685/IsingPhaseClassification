from typing import Any, List
from math import log, sqrt
import numpy as np
import torch
from torch.utils.data import Dataset
from .sampler.metropolis import Metropolis
from .sampler.swedsen_wang import SwendsenWang
from .sampler.wolff import Wolff


class IsingDataset(Dataset):

    def __init__(self, Nx:int, Ny:int, phase:str, data_num:int, Ts:List[float], warmup=100, interval=5) -> None:
        # create list for spin configurations and labels.
        self.x = []
        self.y = []

        for T in Ts:
            # NOTE you can use other samplers ('Metropolis', or 'Wolff').
            sampler = SwendsenWang(
                J=1.0,
                Nx=Nx,
                Ny=Ny,
                T=T
            )

            if phase in ['train', 'val']:
                Tc = 2 * 1.0 / log(sqrt(2) + 1)

            elif phase == 'test':
                Tc = 4 / log(3)
                sampler.nbrs = lambda x, y: [((x+1) % Nx, y), (x, (y+1) % Ny), ((x+1) % Nx, (y+1) % Ny)]
                
            else:
                raise ValueError

            # to equilibrium
            for i in range(warmup):
                sampler.mcstep()

            # sampling
            for i in range(data_num):
                for j in range(interval):
                    sampler.mcstep()
                self.x.append(sampler.spins.flatten())
                self.y.append(1 if T > Tc else 0)
                # print(sampler.spins.sum())

        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        x = torch.tensor(self.x[index]).type(torch.FloatTensor)
        y = torch.tensor(self.y[index]).type(torch.LongTensor)
        return x, y