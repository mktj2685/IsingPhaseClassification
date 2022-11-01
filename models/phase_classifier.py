import torch
import torch.nn as nn


class PhaseClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out


if __name__ == '__main__':
    # check model
    model = PhaseClassifier(20 * 20, 100)
    print(model)

    # check inference
    input = torch.rand(20 * 20)
    output = model(input)
    print(output)