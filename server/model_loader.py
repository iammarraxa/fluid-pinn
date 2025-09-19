import torch
import sys
from pathlib import Path

base = Path(__file__).resolve().parent.parent
sys.path.append(str(base))

models_dir = base/'models'

from model import PINN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIO_WEIGHTS_MAP = {
    'Baseline' : {
        'f' : 'Baseline_net_f.pt',
        'th' : 'Baseline_net_th.pt'
    },

    'Higher_delta' : {
        'f' : 'Higher_delta_net_f.pt',
        'th' : 'Higher_delta_net_th.pt'
    },

    'Higher_lambda' : {
        'f' : 'Higher_lambda_net_f.pt',
        'th' : 'Higher_lambda_net_th.pt'
    },

    'Higher_S' : {
        'f' : 'Higher_S_net_f.pt',
        'th' : 'Higher_S_net_th.pt'
    },

    'Higher_Sq' : {
        'f' : 'Higher_Sq_net_f.pt',
        'th' : 'Higher_Sq_net_th.pt'
    },

    'Stronger_M' : {
        'f' : 'Stronger_M_net_f.pt',
        'th' : 'Stronger_M_net_th.pt'
    }
}

CACHE = {}

def get_nets(scenario : str):

    if scenario not in SCENARIO_WEIGHTS_MAP:
        raise ValueError(f"\nUNKNOWN SCENARIO ==> '{scenario}'\nUSE VALID KEYS ==> {list(SCENARIO_WEIGHTS_MAP.keys())}")
    
    key = (scenario, DEVICE.type)
    if key in CACHE:
        return (CACHE[key][0], CACHE[key][1], DEVICE)
    
    f_path = (models_dir / SCENARIO_WEIGHTS_MAP[scenario]['f']).resolve()
    th_path = (models_dir / SCENARIO_WEIGHTS_MAP[scenario]['th']).resolve()

    if not f_path.exists():
        raise FileNotFoundError(f"Missing weights file: '{f_path}'")
    if not th_path.exists():
        raise FileNotFoundError(f"Missing weights file: '{th_path}'")

    net_f  = PINN([1, 64, 64, 1], activation="sigmoid").to(DEVICE)
    net_th = PINN([1, 64, 64, 1], activation="sigmoid").to(DEVICE)

    state_f = torch.load(f_path, map_location=DEVICE)
    state_th = torch.load(th_path, map_location=DEVICE)

    net_f.load_state_dict(state_f)
    net_th.load_state_dict(state_th)

    net_f.eval()
    net_th.eval()

    CACHE[key] = (net_f, net_th)

    return net_f, net_th, DEVICE