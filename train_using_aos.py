import torch
from model import PINN
from aos import aos
from utils import set_model_weights, get_model_weights
from extract_weights import extract_weights
from plots import plot_results
from params import delta
from aos_loss import thetha_trial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [1, 10, 1]

net_f = PINN(layers).to(device)
net_thetha = PINN(layers).to(device)

weights_f = get_model_weights(net_f)
weights_theta = get_model_weights(net_thetha)

d_f = weights_f.shape[0]
d_theta = weights_theta.shape[0]
d_total = d_f + d_theta

electrons = 20
iterations = 500

final_weights = aos(net_f, net_thetha, device, electrons=electrons, n_of_weights=d_total, d_f=d_f, iterations=iterations, seed=42)
final_weights_f = final_weights[:d_f]
final_weights_thetha = final_weights[d_f:]

set_model_weights(net_f, final_weights_f)
set_model_weights(net_thetha, final_weights_thetha)

extract_weights(net_f)
extract_weights(net_thetha)

with torch.no_grad():
    eta_0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    eta_1 = torch.tensor([[1.0]], dtype=torch.float32).to(device)

    theta_0 = thetha_trial(eta_0, net_thetha).cpu().item()
    theta_1 = thetha_trial(eta_1, net_thetha).cpu().item()

    print(f"Theta(0): {theta_0:.6f}, Expected: {delta}")
    print(f"Theta(1): {theta_1:.6f}, Expected: 1.0")

plot_results(net_f, net_thetha, device)