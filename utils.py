import torch

def gradients(y, x, order=1):
    for _ in range(order):
        y = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return y

def get_model_weights(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

def set_model_weights(model, weights):
    idx = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = weights[idx:idx+numel].view(param.shape).clone()
        idx += numel