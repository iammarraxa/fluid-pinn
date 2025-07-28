# params.py

# Nanoparticle volume fractions
phi1 = 0.01  # Graphene
phi2 = 0.01  # Fe3O4
phi = phi1 + phi2

# Base fluid (Water)
rho_f = 997
mu_f = 0.001003  # kg/m·s
Cp_f = 4180
k_f = 0.6071
sigma_f = 0.005

# Nanoparticles
rho_g = 2250
Cp_g = 2100
k_g = 2500
sigma_g = 1e7

rho_fe3o4 = 5180
Cp_fe3o4 = 670
k_fe3o4 = 9.7
sigma_fe3o4 = 2.5e4

# Effective HNF properties
rho_hnf = (1 - phi) * rho_f + phi1 * rho_g + phi2 * rho_fe3o4
Cp_hnf = ((1 - phi) * rho_f * Cp_f + phi1 * rho_g * Cp_g + phi2 * rho_fe3o4 * Cp_fe3o4) / rho_hnf
k_hnf = (1 - phi) * k_f + phi1 * k_g + phi2 * k_fe3o4
sigma_hnf = (1 - phi) * sigma_f + phi1 * sigma_g + phi2 * sigma_fe3o4
mu_hnf = mu_f / (1 - phi)**2.5

rhoCp_hnf = rho_hnf * Cp_hnf
rhoCp_f = rho_f * Cp_f

# Constants C1–C5
C1 = mu_hnf / mu_f
C2 = rho_hnf / rho_f
C3 = k_hnf / k_f
C4 = rhoCp_hnf / rhoCp_f
C5 = sigma_hnf / sigma_f

# Dimensionless parameters
T1 = 350
T2 = 300
T_inf = 298

delta = (T1 - T_inf) / (T2 - T_inf)

alpha = 0.1
b = 0.5
Sq = alpha / b

nu_f = mu_f / rho_f
Pr = nu_f * rhoCp_f / k_f

B0 = 1.0
M = sigma_f * B0**2 / (b * rho_f)

V0 = 0.1
h = 1.0
S = V0 / (b * h)

# Boundary condition stretch
lam = 0.5  # λ (stretching rate of lower plate)
