# plots.py
import torch
import matplotlib.pyplot as plt
from model import PINN
from aos_loss import f_trial, theta_trial
from utils import gradients
from params import S, Sq, lam, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_and_plot(net_f, net_th, n=400):
    eta = torch.linspace(0, 1, n, device=device).view(-1, 1)
    eta.requires_grad_(True)

    f  = f_trial(eta, net_f)
    f1 = gradients(f, eta, 1)
    f2 = gradients(f, eta, 2)

    th  = theta_trial(eta, net_th)
    th1 = gradients(th, eta, 1)

    # Wall quantities
    cf0 = f2[0].detach().item()      # proportional to skin friction at lower plate
    Nu0 = -th1[0].detach().item()    # Nusselt number proxy

    print(f"f''(0) ~ {cf0:.6f},  -theta'(0) ~ {Nu0:.6f}")

    with torch.no_grad():
        e = eta.detach().cpu().numpy().ravel()
        f_ = f.detach().cpu().numpy().ravel()
        fp = f1.detach().cpu().numpy().ravel()
        th_= th.detach().cpu().numpy().ravel()

    plt.figure()
    plt.plot(e, f_, label='f(η)')
    plt.plot(e, fp, label="f'(η)")
    plt.xlabel('η'); plt.legend(); plt.title('Stream function & velocity')
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(e, th_, label='θ(η)')
    plt.xlabel('η'); plt.legend(); plt.title('Temperature profile')
    plt.tight_layout(); plt.show()

def load_and_plot():
    net_f  = PINN([1, 64, 64, 1], activation="sigmoid").to(device)
    net_th = PINN([1, 64, 64, 1], activation="sigmoid").to(device)
    net_f.load_state_dict(torch.load("net_f.pt", map_location=device))
    net_th.load_state_dict(torch.load("net_th.pt", map_location=device))
    evaluate_and_plot(net_f, net_th)

if __name__ == "__main__":
    load_and_plot()
