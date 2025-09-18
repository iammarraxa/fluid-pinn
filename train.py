import torch
from model import PINN
from utils import chebyshev_lobatto
from aos_loss import f_trial, theta_trial, pde_loss_for_weights
from params import S, Sq, lam, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_f  = PINN([1, 64, 64, 1], activation="sigmoid").to(device)
net_th = PINN([1, 64, 64, 1], activation="sigmoid").to(device)

N = 400
eta = chebyshev_lobatto(N, device=device)
eta.requires_grad_(True)

def residual_loss():
    w_flat = torch.cat([p.view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
    return pde_loss_for_weights(w_flat, net_f, net_th, eta, device)

# Adam warmup
optim = torch.optim.Adam(list(net_f.parameters()) + list(net_th.parameters()), lr=1e-3)
lrdecay = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1500, eta_min=2e-4)
for epoch in range(1500):
    optim.zero_grad(set_to_none=True)
    loss = residual_loss()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(net_f.parameters()) + list(net_th.parameters()), 1.0)
    optim.step()
    lrdecay.step()

    if epoch % 100 == 0:
        print(f"[Adam] {epoch:4d}  loss={loss.item():.3e}")

# LBFGS polish
lbfgs = torch.optim.LBFGS(list(net_f.parameters()) + list(net_th.parameters()), max_iter=800, history_size=50,
                          tolerance_grad=1e-10, tolerance_change=1e-10)

def closure():
    lbfgs.zero_grad(set_to_none=True)
    l = residual_loss()
    l.backward()
    return l

lbfgs.step(closure)
print("Training complete. Saving models...")

torch.save(net_f.state_dict(), "net_f.pt")
torch.save(net_th.state_dict(), "net_th.pt")
print("Saved net_f.pt and net_th.pt")
