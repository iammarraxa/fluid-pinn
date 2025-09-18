# baseline_rk4.py
# Purpose: Provide a basic RK4 + shooting method baseline for the paper's ODEs
# so you can compare PINN wall metrics against a classical solver.
#
# Note: This is a simple implementation (no SciPy). For production-grade
# accuracy, SciPy's solve_bvp is recommended. This script aims for decent
# agreement to check trends and magnitudes.
import math
import torch
from params import A4, M_hat, inv_Pr_hat, S, Sq, lam, configure

device = torch.device('cpu')

def rhs(eta, Y):
    # Y = [f, f', f'', f''', theta, theta']
    f, f1, f2, f3, th, th1 = Y
    # f4' from momentum ODE:
    # A4 f'''' + f f''' - f' f'' - (Sq/2)(3 f'' + eta f''') - M_hat f'' = 0
    f4 = ( - f * f3 + f1 * f2 + 0.5*Sq * (3.0*f2 + eta*f3) + M_hat * f2 ) / A4
    # theta'' from energy:
    # inv_Pr_hat * theta'' + f theta' - (Sq/2) eta theta' = 0
    # => theta'' = - Pr_hat * ( f theta' - (Sq/2) eta theta' )
    Pr_hat = 1.0 / inv_Pr_hat
    th2 = - Pr_hat * ( f * th1 - 0.5*Sq * eta * th1 )
    return torch.stack([f1, f2, f3, f4, th1, th2]).type_as(Y)

def rk4_step(eta, Y, h):
    k1 = rhs(eta, Y)
    k2 = rhs(eta + 0.5*h, Y + 0.5*h*k1)
    k3 = rhs(eta + 0.5*h, Y + 0.5*h*k2)
    k4 = rhs(eta + h,     Y + h*k3)
    return Y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(f2_0, f3_0, th1_0, N=200):
    # Initial conditions at eta=0
    eta0 = 0.0
    f0, f1_0 = S, lam
    from params import delta
    th0 = delta
    Y = torch.tensor([f0, f1_0, f2_0, f3_0, th0, th1_0], dtype=torch.float64, device=device)
    h = 1.0 / N
    eta = eta0
    for _ in range(N):
        Y = rk4_step(eta, Y, h)
        eta += h
    return Y  # Y at eta=1

def shoot(targets, guess, tol=1e-6, maxit=30):
    # targets: ( f(1), f'(1), theta(1) ) = (Sq/2, 0, 1)
    t1, t2, t3 = targets
    f2, f3, th1 = guess
    err = float('inf')
    # simple decoupled secant-like updates (not robust but serviceable)
    for it in range(maxit):
        Y1 = integrate(f2, f3, th1)
        r = torch.tensor([Y1[0]-t1, Y1[1]-t2, Y1[4]-t3], dtype=torch.float64)
        err = torch.linalg.norm(r).item()
        # print(f"iter {it} residual {err:.3e}")
        if err < tol:
            break
        # finite-diff sensitivities
        eps = 1e-4
        J = torch.zeros(3,3, dtype=torch.float64)
        for j, (df2, df3, dth1) in enumerate([(eps,0,0),(0,eps,0),(0,0,eps)]):
            Y2 = integrate(f2+df2, f3+df3, th1+dth1)
            r2 = torch.tensor([Y2[0]-t1, Y2[1]-t2, Y2[4]-t3], dtype=torch.float64)
            J[:,j] = (r2 - r) / (eps if j==0 else (eps if j==1 else eps))
        # solve J * dx = -r
        try:
            dx = torch.linalg.solve(J, -r)
        except RuntimeError:
            # fallback: gradient step
            dx = -0.1 * r
        f2 += dx[0].item()
        f3 += dx[1].item()
        th1 += dx[2].item()
    return (f2, f3, th1), err

def main():
    # Optionally configure here if needed (example):
    # configure(alpha=0.6, V0=0.08, lam=0.4, B0=0.8, T1=301.0, T2=300.0, T_inf=298.0, phi1=0.01, phi2=0.01)
    targets = (0.5*Sq, 0.0, 1.0)
    guess = (0.0, 0.0, 0.0)
    (f2_0, f3_0, th1_0), err = shoot(targets, guess, tol=1e-6, maxit=25)
    print(f"Guessed ICs: f''(0)={f2_0:.6f}, f'''(0)={f3_0:.6f}, theta'(0)={th1_0:.6f}, residual={err:.2e}")

if __name__ == "__main__":
    main()
