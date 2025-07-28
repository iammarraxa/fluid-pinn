import matplotlib.pyplot as plt
import torch

def plot_results(net_f, net_thetha, device):
    eta = torch.linspace(0, 1, 200).view(-1, 1).to(device).requires_grad_()
    f = net_f(eta).detach().cpu().numpy()
    thetha = net_thetha(eta).detach().cpu().numpy()
    eta = eta.detach().cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(eta, f, label='f(η)')
    plt.title('Vel Profile')
    plt.xlabel('η')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(eta, thetha, label='θ(η)', color='red')
    plt.title('Temp Profile')
    plt.xlabel('η')
    plt.grid()

    plt.tight_layout()
    plt.show()