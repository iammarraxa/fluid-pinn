import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent
profiles_dir = base / "profiles"

baseline = np.load(profiles_dir / "Baseline_profiles.npz")
eta0   = baseline["eta"]
fp0    = baseline["fp"]
th0    = baseline["theta"]


# Group scenarios by parameter family
groups = {
    "Sq": ["Baseline", "Low_Sq_0p2", "High_Sq_1p2", "VeryHigh_Sq_2p0"],
    "S": ["No_Suction_S_0", "High_Suction_S_0p16", "VeryHigh_Suction_S_0p32"],
    "lambda": ["Low_lambda_0p1", "Baseline", "High_lambda_0p8"],
    "M": ["Weak_M_B0_0p2", "Baseline", "Strong_M_B0_1p6", "VeryStrong_M_B0_3p0"],
    "delta": ["Low_delta_1p2", "Mid_delta_2p5", "High_delta_5p0"]
}

colors = {
    "Sq": "tab:blue",
    "S": "tab:orange",
    "lambda": "tab:green",
    "M": "tab:red",
    "delta": "tab:purple"
}

order = ["Sq", "S", "lambda", "M", "delta"]

styles = ["-", "--", "-.", ":"]  # cycle line styles

def safe_load(name: str):
    path = profiles_dir / f"{name}_profiles.npz"
    if not path.exists():
        print(f"[warn] missing profile: {name}")
        return None
    return np.load(path)

def plot_family(ax, family, ykey, ylabel, title_suffix="", show_delta=False, zoom=None):
    ax.set_xlabel("η")
    ax.set_ylabel(ylabel)
    col = colors[family]
    scens = groups[family]
    for i, sc in enumerate(scens):
        dat = safe_load(sc)
        if dat is None: 
            continue
        style = styles[i % len(styles)]
        y = dat[ykey]
        if show_delta:
            ref = fp0 if ykey == "fp" else th0
            y = y - ref
        ax.plot(dat["eta"], y, style, color=col, lw=1.8, label=sc)
    if zoom is not None:
        ax.set_xlim(0, zoom)
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_title(f"{family} {title_suffix}", fontsize=11)
    ax.legend(fontsize=8)

def figure_absolute(zoom=None):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axs = axs.ravel()
    for k, fam in enumerate(order):
        ax = axs[k]
        plot_family(ax, fam, "fp", "f'(η)", "— Velocity", zoom=zoom)
    ax = axs[-1]
    for fam in order:
        for i, sc in enumerate(groups[fam]):
            dat = safe_load(sc)
            if dat is None: 
                continue
            style = styles[i % len(styles)]
            ax.plot(dat["eta"], dat["theta"], style, color=colors[fam], lw=1.6, label=f"{fam}:{sc}")
    ax.set_xlabel("η"); ax.set_ylabel("θ(η)")
    if zoom is not None: ax.set_xlim(0, zoom)
    ax.set_title("Temperature (all families)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.7)
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        fam = lab.split(":")[0]
        if fam not in seen:
            seen.add(fam); h2.append(h); l2.append(fam)
    ax.legend(h2, l2, title="Families", fontsize=8)

    fig.suptitle("Absolute profiles by family", fontsize=13, y=1.02)
    fig.savefig(base / "panels_absolute.png", dpi=300)
    plt.show()


def figure_deltas(zoom=None):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axs = axs.ravel()
    for k, fam in enumerate(order):
        ax = axs[k]
        plot_family(ax, fam, "fp", "Δ f'(η) vs Baseline", "— Δ Velocity", show_delta=True, zoom=zoom)
    ax = axs[-1]
    for fam in order:
        for i, sc in enumerate(groups[fam]):
            if sc == "Baseline":
                continue
            dat = safe_load(sc)
            if dat is None: 
                continue
            style = styles[i % len(styles)]
            ax.plot(dat["eta"], dat["theta"] - th0, style, color=colors[fam], lw=1.6, label=f"{fam}:{sc}")
    ax.set_xlabel("η"); ax.set_ylabel("Δ θ(η) vs Baseline")
    if zoom is not None: ax.set_xlim(0, zoom)
    ax.set_title("Temperature Δ (all families)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.7)
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels):
        fam = lab.split(":")[0]
        if fam not in seen:
            seen.add(fam); h2.append(h); l2.append(fam)
    ax.legend(h2, l2, title="Families", fontsize=8)

    fig.suptitle("Differences from baseline", fontsize=13, y=1.02)
    fig.savefig(base / "panels_deltas.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    figure_absolute(zoom=None)
    figure_deltas(zoom=None)
    figure_absolute(zoom=0.2)
    figure_deltas(zoom=0.2)
