# run_sweep.py
import json, csv, torch
from pathlib import Path
from model import PINN
from utils import chebyshev_lobatto, gradients
import params as P
from params import configure
from aos_loss import f_trial, theta_trial, pde_loss_for_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = Path(__file__).parent

N = 300 
ADAM_STEPS = 600
LBFGS_STEPS = 300

def train_once(net_f, net_th, eta, adam_steps=600, lbfgs_steps=300):
    # Adam
    optim = torch.optim.Adam(list(net_f.parameters()) + list(net_th.parameters()), lr=3e-3)
    for epoch in range(adam_steps):
        optim.zero_grad(set_to_none=True)
        w_flat = torch.cat([p.view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
        loss = pde_loss_for_weights(w_flat, net_f, net_th, eta, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net_f.parameters()) + list(net_th.parameters()), 1.0)
        optim.step()
    # LBFGS
    lbfgs = torch.optim.LBFGS(list(net_f.parameters()) + list(net_th.parameters()), max_iter=lbfgs_steps, history_size=50,
                              tolerance_grad=1e-9, tolerance_change=1e-9)
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        w_flat = torch.cat([p.view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
        l = pde_loss_for_weights(w_flat, net_f, net_th, eta, device)
        l.backward()
        return l
    lbfgs.step(closure)

def wall_metrics(net_f, net_th):
    eta = torch.linspace(0, 1, 401, device=device).view(-1,1)
    eta.requires_grad_(True)
    f  = f_trial(eta, net_f)
    th = theta_trial(eta, net_th)
    f2 = gradients(f, eta, 2)
    th1 = gradients(th, eta, 1)
    cf0 = f2[0].detach().item()
    Nu0 = -th1[0].detach().item()
    return cf0, Nu0

def main():
    scenarios = json.loads((base / "scenarios.json").read_text())
    out_csv = base / "sweep_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","Sq","S","lambda","Mhat","delta","phi1","phi2","fpp0","-thetap0","final_loss"])

        for sc in scenarios:
            # apply scenario
            vals = configure(**sc)
            print(f"=== Scenario: {sc['name']} | Sq={P.Sq:.4f}")
            # build fresh nets and collocation
            net_f  = PINN([1, 64, 64, 1], activation="sigmoid").to(device)
            net_th = PINN([1, 64, 64, 1], activation="sigmoid").to(device)
            eta = chebyshev_lobatto(N, device=device)
            eta.requires_grad_(True)

            train_once(net_f, net_th, eta, adam_steps=ADAM_STEPS, lbfgs_steps=LBFGS_STEPS)

            # final loss
            
            wf = torch.cat([p.view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
            fin_loss = pde_loss_for_weights(wf, net_f, net_th, eta, device).item()
            fpp0, nuth0 = wall_metrics(net_f, net_th)

            eta_plot = torch.linspace(0, 1, 201, device=device).view(-1,1)
            eta_plot.requires_grad_(True)
            f_vals  = f_trial(eta_plot, net_f).detach().cpu().numpy().flatten()
            fp_vals = gradients(f_trial(eta_plot, net_f), eta_plot, 1).detach().cpu().numpy().flatten()
            th_vals = theta_trial(eta_plot, net_th).detach().cpu().numpy().flatten()
            import numpy as np, os
            os.makedirs(base / "profiles", exist_ok=True)
            np.savez(base / f"profiles/{sc['name']}_profiles.npz",
                    eta=eta_plot.detach().cpu().numpy().flatten(),
                    f=f_vals, fp=fp_vals, theta=th_vals)

            # write CSV line
            w.writerow([
                sc["name"], P.Sq, P.S, P.lam, P.M_hat, P.delta, P.phi1, P.phi2,
                fpp0, nuth0, fin_loss
            ])
            print(f" -> f''(0)={fpp0:.5f}, -theta'(0)={nuth0:.5f}, loss={fin_loss:.2e}")

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
