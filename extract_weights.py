import torch
from model import PINN

# def extract_weights(model):
#     with torch.no_grad():
#         l1_w = model.hidden_layers[0].weight.detach().cpu().numpy()
#         l1_b = model.hidden_layers[0].bias.detach().cpu().numpy()
#         out_w = model.output_layer.weight.detach().cpu().numpy()

#         print("    w(f_i)       c(f_1)       a(f_i)")
#         for i in range(l1_w.shape[0]):
#             w = l1_w[i, 0]
#             c = l1_b[i]
#             a = out_w[0, i]
#             print(f"{w:>12.8f} {c:>12.8f} {a:>12.8f}")

def print_first_layer(model: PINN, name="net"):
    
    first = None
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            first = m
            break
    if first is None:
        print("No Linear layer found.")
        return
    W = first.weight.detach().cpu().numpy()
    b = first.bias.detach().cpu().numpy()
    print(f"== {name} first layer weights shape={W.shape}, bias shape={b.shape}")

    for i in range(min(8, W.shape[0])):
        print(f"row[{i}]: w={W[i,0]: .6f}, b={b[i]: .6f}")

if __name__ == "__main__":
    net_f  = PINN([1, 32, 32, 1], activation="sigmoid")
    net_th = PINN([1, 32, 32, 1], activation="sigmoid")
    print_first_layer(net_f, "net_f")
    print_first_layer(net_th, "net_th")