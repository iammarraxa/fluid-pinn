import torch
from utils import gradients, set_model_weights
from params import A4, M_hat, inv_Pr_hat, S, Sq, delta, lam


def f_trial(eta, net_f):
    NN_out = net_f(eta)

    # Basis that satisfies f(0), f'(0), f(1), f'(1)
    H0 = 1 - 3*eta**2 + 2*eta**3
    H1 = eta - 2*eta**2 + eta**3
    H2 = 3*eta**2 - 2*eta**3
    H3 = -eta**2 + eta**3

    return H0*S + H1*lam + H2*(Sq/2.0) + H3*0.0 + (eta**2)*(1 - eta)**2 * NN_out


def theta_trial(eta, net_thetha):
    base = delta*(1 - eta) + eta
    core = net_thetha(eta)
    return base + eta * (1 - eta) * core


def pde_loss_for_weights(w_flat, net_f, net_th, eta, device):
    n_f = sum(p.numel() for p in net_f.parameters())
    wf = w_flat[:n_f]
    wt = w_flat[n_f:]

    set_model_weights(net_f, wf)
    set_model_weights(net_th, wt)

    eta = eta.to(device, dtype=next(net_f.parameters()).dtype)
    eta.requires_grad_(True)

    f  = f_trial(eta, net_f)
    f1 = gradients(f, eta, 1)
    f2 = gradients(f, eta, 2)
    f3 = gradients(f, eta, 3)
    f4 = gradients(f, eta, 4)

    th  = theta_trial(eta, net_th)
    th1 = gradients(th, eta, 1)
    th2 = gradients(th, eta, 2)

    # Momentum
    r_mom = A4 * f4 + f * f3 - f1 * f2 - 0.5*Sq * (3.0*f2 + eta * f3) - M_hat * f2

    # Energy
    r_eng = inv_Pr_hat * th2 + f * th1 - 0.5*Sq * eta * th1


    mom_mse = torch.mean(r_mom**2)
    eng_mse = torch.mean(r_eng**2)

    loss = mom_mse + 2.0 * eng_mse
    return loss