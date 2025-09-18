# train_using_aos.py
import torch
from model import PINN
from utils import get_model_weights, set_model_weights, chebyshev_lobatto, count_params
from aos import AOS
from aos_loss import pde_loss_for_weights
from params import S, Sq, lam, delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Networks (sigmoid for paper-style smooth derivatives)
net_f  = PINN([1, 32, 32, 1], activation="sigmoid").to(device)
net_th = PINN([1, 32, 32, 1], activation="sigmoid").to(device)

# Collocation points
N = 300
eta = chebyshev_lobatto(N, device=device)

# Flattened parameter vector
w0 = torch.cat([p.detach().view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
dim = w0.numel()

# Objective wrapper for AOS (no grad)
def objective_no_grad(w):
    with torch.no_grad():
        return pde_loss_for_weights(w, net_f, net_th, eta, device).item()

# AOS bounds
lo = -5.0 * torch.ones(dim, device=device)
hi =  5.0 * torch.ones(dim, device=device)

# Run AOS
aos = AOS(dim=dim, pop_size=30, pr_init=0.5, pr_final=0.1, iters=1500, device=device)
w_best, f_best = aos.run((lo, hi), objective_no_grad)

# Load best weights into networks
set_model_weights(net_f,  w_best[:count_params(net_f)])
set_model_weights(net_th, w_best[count_params(net_f):])

# Gradient refinement: Adam -> LBFGS
eta.requires_grad_(True)

def loss_fn():
    from aos_loss import pde_loss_for_weights
    w_flat = torch.cat([p.view(-1) for p in list(net_f.parameters()) + list(net_th.parameters())])
    return pde_loss_for_weights(w_flat, net_f, net_th, eta, device)

# Adam
optim = torch.optim.Adam(list(net_f.parameters()) + list(net_th.parameters()), lr=3e-3)
for epoch in range(1000):
    optim.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(net_f.parameters()) + list(net_th.parameters()), max_norm=1.0)
    optim.step()
    if epoch % 100 == 0:
        print(f"[Adam] {epoch:4d}  loss={loss.item():.3e}")

# LBFGS
lbfgs = torch.optim.LBFGS(list(net_f.parameters()) + list(net_th.parameters()), max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-9, history_size=50)

def closure():
    lbfgs.zero_grad(set_to_none=True)
    l = loss_fn()
    l.backward()
    return l

lbfgs.step(closure)
print("Refinement complete.")

# Save models
torch.save(net_f.state_dict(), "net_f.pt")
torch.save(net_th.state_dict(), "net_th.pt")
print("Saved net_f.pt and net_th.pt")
