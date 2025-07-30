import matplotlib.pyplot as plt
import torch
from aos_loss import f_trial, thetha_trial
from params import *

def plot_results(net_f, net_thetha, device):
    eta = torch.linspace(0, 1, 200).view(-1, 1).to(device).requires_grad_()
    # f = net_f(eta).detach().cpu().numpy()
    # thetha = net_thetha(eta).detach().cpu().numpy()
    f = f_trial(eta, net_f).detach().cpu().numpy()
    thetha = thetha_trial(eta, net_thetha).detach().cpu().numpy()
    eta = eta.detach().cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(eta, f, label='f(η)')
    plt.plot([0], [S], 'ko', label='f(0)')
    plt.plot([1], [Sq/2], 'ks', label='f(1)')
    plt.title('Vel Profile')
    plt.xlabel('η')
    plt.ylabel('f')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eta, thetha, label='θ(η)', color='red')
    plt.plot([0], [delta], 'ko', label='θ(0)')
    plt.plot([1], [1.0], 'ks', label='θ(1)')
    plt.title('Temp Profile')
    plt.xlabel('η')
    plt.ylabel('θ')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()