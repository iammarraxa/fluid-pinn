# Nanoparticle volume fractions
phi1 = 0.01  # Graphene
phi2 = 0.01  # Fe3O4
phi = phi1 + phi2

# Base fluid (Water)
rho_f = 997.0
mu_f = 1.003e-3
Cp_f = 4180.0
k_f = 0.6071
sigma_f = 0.005

# Nanoparticles
rho_g = 2250.0                  ##############
Cp_g = 2100.0                      #GRAPHENE#
k_g = 2500.0
sigma_g = 1.0e7                 ##############

rho_fe3o4 = 5180.0              ##############
Cp_fe3o4 = 670.0                    #FE304#
k_fe3o4 = 6.0
sigma_fe3o4 = 2.5e4             ##############

# Flow & field parameters
alpha = 0.8
b = 1.0
B0 = 1.0
V0 = 0.1
h = 1.0
lam = 0.5

# Boundary cond. temps
T1 = 303.0
T2 = 300.0
T_inf = 298.0

#HNF properties
def brinkman_mu(mu_base: float, phi_total: float) -> float:
    return mu_base / ((1.0 - phi_total) ** 2.5)

def maxwell_mix_k(k_base: float, k_part: float, phi_part: float) -> float:
    if phi_part <= 0.0:
        return k_base
    num = k_part + 2.0 * k_base - 2.0 * phi_part * (k_base - k_part)
    den = k_part + 2.0 * k_base + phi_part * (k_base - k_part)
    return k_base * (num / den)

def sequential_maxwell(k_base: float, k_p1: float, phi1: float, k_p2: float, phi2: float) -> float:
    k_mid = maxwell_mix_k(k_base, k_p1, phi1)
    k_eff = maxwell_mix_k(k_mid,  k_p2, phi2)
    return k_eff

# Effective density and heat capacity
rho_hnf = (1.0 - phi) * rho_f + phi1 * rho_g + phi2 * rho_fe3o4
rhoCp_f = rho_f * Cp_f
rhoCp_hnf = (1.0 - phi) * rhoCp_f + phi1 * (rho_g * Cp_g) + phi2 * (rho_fe3o4 * Cp_fe3o4)

# Thermal conductivity
k_hnf = sequential_maxwell(k_f, k_g, phi1, k_fe3o4, phi2)

# Electrical conductivity
sigma_hnf = sequential_maxwell(sigma_f, sigma_g, phi1, sigma_fe3o4, phi2)

# Dynamic viscosity
mu_hnf = brinkman_mu(mu_f, phi)

# Kinematic viscosities
nu_f = mu_f / rho_f
nu_hnf = mu_hnf / rho_hnf

# Prandtl nums
Pr_f = nu_f * rhoCp_f / k_f
Pr_hnf = nu_hnf * rhoCp_hnf / k_hnf

# Magnetic params
M_base = sigma_f * (B0**2) / (b * rho_f)
M_eff = sigma_hnf * (B0**2) / (b * rho_hnf)

# Squeezing
Sq = alpha / b

# Suction
S = V0 / (b * h)

# Temp ratio
delta = (T1 - T_inf) / (T2 - T_inf)

#9 Constants C1–C5
# C1 = mu_hnf / mu_f
# C2 = rho_hnf / rho_f
# C3 = k_hnf / k_f
# C4 = rhoCp_hnf / rhoCp_f
# C5 = sigma_hnf / sigma_f

# C1 / C2
A4 = (mu_hnf * rho_f) / (mu_f * rho_hnf)

# Magnetic term uses effective magnetic parameter
M_hat = M_eff

# Energy equation
inv_Pr_hat = 1.0 / Pr_hnf

__all__ = [
    "phi1", "phi2", "phi",
    "rho_f", "mu_f", "Cp_f", "k_f", "sigma_f",
    "rho_g", "Cp_g", "k_g", "sigma_g",
    "rho_fe3o4", "Cp_fe3o4", "k_fe3o4", "sigma_fe3o4",
    "rho_hnf", "mu_hnf", "k_hnf", "sigma_hnf", "rhoCp_hnf",
    "nu_f", "nu_hnf", "Pr_f", "Pr_hnf",
    "A4", "M_hat", "inv_Pr_hat",
    "S", "Sq", "delta", "lam",
]


# ------------------ Scenario configuration API ---------------------------
def configure(**kw):
    globals().update(kw)

    global phi, rho_hnf, rhoCp_f, rhoCp_hnf, mu_hnf, k_hnf, sigma_hnf
    global nu_f, nu_hnf, Pr_f, Pr_hnf, M_base, M_eff, Sq, S, delta
    global A4, M_hat, inv_Pr_hat

    phi = phi1 + phi2

    rho_hnf   = (1.0 - phi) * rho_f   + phi1 * rho_g   + phi2 * rho_fe3o4
    rhoCp_f   = rho_f * Cp_f
    rhoCp_hnf = (1.0 - phi) * rhoCp_f + phi1 * (rho_g * Cp_g) + phi2 * (rho_fe3o4 * Cp_fe3o4)

    mu_hnf = brinkman_mu(mu_f, phi)
    k_hnf = sequential_maxwell(k_f, k_g, phi1, k_fe3o4, phi2)
    sigma_hnf = sequential_maxwell(sigma_f, sigma_g, phi1, sigma_fe3o4, phi2)

    nu_f   = mu_f / rho_f
    nu_hnf = mu_hnf / rho_hnf

    Pr_f   = nu_f   * (rhoCp_f)   / k_f
    Pr_hnf = nu_hnf * (rhoCp_hnf) / k_hnf

    M_base = sigma_f   * (B0**2) / (b * rho_f)
    M_eff  = sigma_hnf * (B0**2) / (b * rho_hnf)

    Sq = alpha / b
    S  = V0 / (b * h)

    delta = (T1 - T_inf) / (T2 - T_inf)

    A4 = (mu_hnf * rho_f) / (mu_f * rho_hnf)
    M_hat = M_eff
    inv_Pr_hat = 1.0 / Pr_hnf

    return {k: globals()[k] for k in __all__}