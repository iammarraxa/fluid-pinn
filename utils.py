import torch

def gradients(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return y

def get_model_weights(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def set_model_weights(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        with torch.no_grad():
            p.copy_(flat[idx:idx+n].view_as(p))
        idx += n
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def chebyshev_lobatto(N, device=None, dtype=torch.float32):
    j = torch.arange(N, device=device, dtype=dtype)
    eta = 0.5 * (1.0 - torch.cos(torch.pi * j / (N - 1)))
    return eta.view(-1, 1)