import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers, activation="sigmoid"):
        super().__init__()
        acts = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }
        self.act = acts[activation]
        net = []
        for i in range(len(layers)-1):
            lin = nn.Linear(layers[i], layers[i+1])
            # Xavier init tailored to sigmoid/tanh
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            net.append(lin)
            if i < len(layers)-2:
                net.append(self.act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
