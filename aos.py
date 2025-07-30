import torch
import numpy as np
from aos_loss import loss
from utils import get_model_weights

def aos(net_f, net_thetha, device, electrons=20, n_of_weights=100, d_f=100, iterations=100, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


    X = np.random.uniform(-1, 1, (electrons, n_of_weights))
    # E = np.array([loss(model, torch.tensor(x, dtype=torch.float32).to(device), device) for x  in X])
    E = np.array([
    loss(net_f, net_thetha, torch.tensor(x[:d_f], dtype=torch.float32).to(device), torch.tensor(x[d_f:], dtype=torch.float32).to(device), device) for x in X])
    PR = 0.4

    LE_idx = np.argmin(E)
    LE = E[LE_idx]
    X_LE = X[LE_idx]

    
    alpha = np.random.rand(electrons)
    beta = np.random.rand(electrons)
    gamma = np.random.rand(electrons)

    for k in range(iterations):
        BS = np.mean(X, axis=0)
        BE = np.mean(E)
        r = np.random.uniform(0, 1, (electrons, n_of_weights))

        for i in range(electrons):
            phi = np.random.rand()
            Xi = X[i]
            Ei = E[i]

            if phi >= PR:
                if Ei >= BE:
                    Xi_new = Xi + (alpha[i] * (beta[i]*LE - gamma[i]*BS)) / (k+1)
                else:
                    Xi_new = Xi + (alpha[i] * (beta[i]*LE -gamma[i]*BS))
            else:
                Xi_new = Xi + r[i]

            Xi_new = np.clip(Xi_new, -5, 5)
            new_loss = loss(net_f, net_thetha, torch.tensor(Xi_new[:d_f], dtype=torch.float32).to(device), torch.tensor(Xi_new[d_f:], dtype=torch.float32).to(device), device)

            if new_loss < Ei:
                X[i] = Xi_new
                E[i] = new_loss
                if new_loss < LE:
                    LE = new_loss
                    X_LE = Xi_new
        
        if k % 10 == 0:
            print(f"Iteration {k} | Best Loss : {LE:.6f}")
        

    return torch.tensor(X_LE, dtype=torch.float32).to(device)


