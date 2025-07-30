import torch
from utils import gradients, set_model_weights
from params import C1, C2, C3, C4, C5, Pr, M, S, Sq, delta, lam


def f_trial(eta, net_f):
    NN_out = net_f(eta)

    # Basis that satisfies f(0), f'(0), f(1), f'(1)
    H0 = 1 - 3*eta**2 + 2*eta**3
    H1 = eta - 2*eta**2 + eta**3
    H2 = 3*eta**2 - 2*eta**3
    H3 = -eta**2 + eta**3

    return H0 * S + H1 * lam + H2 * (Sq / 2) + H3 * 0 + eta**2 * (1 - eta)**2 * NN_out


def thetha_trial(eta, net_thetha):
    NN_out = net_thetha(eta)
    return delta + (1 - delta) * eta + eta**2 * (1 - eta)**2 * NN_out


def loss(net_f, net_thetha, weights_f, weights_thetha, device):
    set_model_weights(net_f, weights_f)
    set_model_weights(net_thetha, weights_thetha)
    # model.eval()

    N = 100
    eta = torch.linspace(0, 1, N).view(-1, 1).to(device)
    eta = eta.requires_grad_()


    # output = model(eta)

    # f = net_f(eta)
    f = f_trial(eta, net_f)
    f1 = gradients(f, eta, 1)
    f2 = gradients(f, eta, 2)
    f3 = gradients(f, eta, 3)
    f4 = gradients(f, eta, 4)

    thetha = thetha_trial(eta, net_thetha)
    thetha1 = gradients(thetha, eta, 1)
    thetha2 = gradients(thetha, eta, 2)


    eq1 = (C1/C2)*f4 + f*f3 - f1*f2 - (Sq/2)*(3*f2 + eta*f3) - (C5/C2)*M*f2 
    eq2 = (1/Pr)*(C3/C4)*thetha2 + f*thetha1 - (Sq/2)*eta*thetha1

    loss_eq = (torch.mean(eq1**2) + torch.mean(eq2**2))/ 1e6


    eta_0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    eta_1 = torch.tensor([[1.0]], requires_grad=True).to(device)

    # f_0 = net_f(eta_0)
    # f_1 = net_f(eta_1)
    # f1_0 = gradients(f_0, eta_0, 1)
    # f1_1 = gradients(f_1, eta_1, 1)

    # thetha_0 = net_thetha(eta_0)
    # thetha_1 = net_thetha(eta_1)

    f_0 = f_trial(eta_0, net_f)
    f_1 = f_trial(eta_1, net_f)
    f1_0 = gradients(f_0, eta_0, 1)
    f1_1 = gradients(f_1, eta_1, 1)

    thetha_0 = thetha_trial(eta_0, net_thetha)
    thetha_1 = thetha_trial(eta_1, net_thetha)


    
    loss_bc = (f_0 - S)**2 + (f_1 - (Sq/2))**2 + (f1_0 - lam)**2 + (f1_1)**2 + (thetha_0 - delta)**2 + (thetha_1 - 1)**2

    # w_bc = loss_eq.item() / loss_bc.item()
    w_bc = 10000
    loss = loss_eq + w_bc * torch.mean(loss_bc)

    # loss = loss_eq

    if torch.rand(1).item() < 0.001:  # Print ~1% of the time to avoid spam
        print(f"f(0): {f_0.item():.4f}, f'(0): {f1_0.item():.4f}, f(1): {f_1.item():.4f}, f'(1): {f1_1.item():.4f}")
        print(f"theta(0): {thetha_0.item():.4f}, theta(1): {thetha_1.item():.4f}")
        print(f"PDE Loss: {loss_eq.item():.3f}, BC Loss: {loss_bc.item():.3f}, Total: {loss.item():.3f}")
        # print(f"PDE Loss: {loss_eq.item():.3f}")


    return loss.item()
