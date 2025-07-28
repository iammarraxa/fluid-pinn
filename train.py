import torch
import torch.nn as nn
import torch.optim as optim

from model import PINN
from utils import gradients
from params import C1, C2, C3, C4, C5, Pr, M, S, Sq, delta, lam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers = [1, 32, 32, 32, 1]
net_f = PINN(layers).to(device)
net_thetha = PINN(layers).to(device)

N = 100
eta = torch.linspace(0, 1, N).view(-1, 1).to(device)
eta.requires_grad = True


lr = 1e-3

optimizer = optim.Adam(list(net_f.parameters()) + list(net_thetha.parameters()), lr)
epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()

    f = net_f(eta)
    f1 = gradients(f, eta, 1)
    f2 = gradients(f, eta, 2)
    f3 = gradients(f, eta, 3)
    f4 = gradients(f, eta, 4)

    thetha = net_thetha(eta)
    thetha1 = gradients(thetha, eta, 1)
    thetha2 = gradients(thetha, eta, 2)

    eq1 = (C1/C2)*f4 + f*f3 - f1*f2 - (Sq/2)*(3*f2 + eta*f3) - (C5/C2)*M*f2 
    eq2 = (1/Pr)*(C3/C4)*thetha2 + f*thetha1 - (Sq/2)*eta*thetha1

    loss_eq = torch.mean(eq1**2) + torch.mean(eq2**2)

    eta_0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    eta_1 = torch.tensor([[1.0]], requires_grad=True).to(device)

    f_0 = net_f(eta_0)
    f_1 = net_f(eta_1)
    f1_0 = gradients(f_0, eta_0)
    f1_1 = gradients(f_1, eta_1)

    thetha_0 = net_thetha(eta_0)
    thetha_1 = net_thetha(eta_1)

    loss_bc = (f_0 - S)**2 + (f_1 - (Sq/2))**2 + (f1_0 - lam)**2 + (f1_1)**2 + (thetha_0 - delta)**2 + (thetha_1 - 1)**2

    loss = loss_eq + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"EPOCH {epoch} | LOSS = {loss.item():.4e}")
        