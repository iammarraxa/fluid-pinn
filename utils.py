import torch

def gradients(y, x, order=1):
    for i in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return y