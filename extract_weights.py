import numpy as np
import pandas as pd
import torch

def extract_weights(model):
    with torch.no_grad():
        l1_w = model.hidden_layers[0].weight.detach().cpu().numpy()
        l1_b = model.hidden_layers[0].bias.detach().cpu().numpy()
        out_w = model.output_layer.weight.detach().cpu().numpy()

        print("    w(f_i)       c(f_1)       a(f_i)")
        for i in range(l1_w.shape[0]):
            w = l1_w[i, 0]
            c = l1_b[i]
            a = out_w[0, i]
            print(f"{w:>12.8f} {c:>12.8f} {a:>12.8f}")

    # df = pd.DataFrame({
    #     'w(f_i)': l1_w[:, 0],
    #     'c(f_i)': l1_b,
    #     'a(f_i)': l2_w[0]
    # })
    # df.to_csv("optimized_weights.csv", index=False)
    # print("\nSaved to optimized_weights.csv")