import torch
import numpy as np
import typing

from model_loader import get_nets
from aos_loss import f_trial, theta_trial
from utils import gradients

def compute_profiles(scenario : str, n_points = 401) -> dict :
    net_f, net_th, device = get_nets(scenario)

    eta = torch.linspace(0.0, 1.0, n_points, device=device).view(-1,1)
    eta.requires_grad_(True)

    f = f_trial(eta, net_f)
    fp = gradients(f, eta, 1)
    theta = theta_trial(eta, net_th)

    f2 = gradients(f, eta, 2)
    th1 = gradients(theta, eta, 1)

    eta_list = eta.detach().squeeze(1).cpu().numpy().tolist()
    fp_list = fp.detach().squeeze(1).cpu().numpy().tolist()
    theta_list = theta.detach().squeeze(1).cpu().numpy().tolist()

    cf = f2[0].detach().item()
    Nu = (-th1[0]).detach().item()

    return {"scenario" : scenario,
            "η" : eta_list,
            "fp" : fp_list,
            "θ" : theta_list,
            "wall" : {"cf" : cf, "Nu" : Nu}}