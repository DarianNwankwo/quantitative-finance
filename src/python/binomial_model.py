import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import factorial, sqrt
from scipy.special import comb


def binomial_model(S: int, u: float, v: float, p: float, t: int):
    """
    S: strike price
    u: factor of increase
    v: factor of decrease
    p: probability of increase
    """
    leaf_nodes = np.array([0]*(t+1))
    probs = []

    # 0, 1, 2, 3, 4
    for h in np.arange(t+1):
        val = (u**h * v**(t-h)) 
        leaf_nodes[h] = val * S
        prob = comb(t, h) * (p**h) * ((1-p)**(t-h))
        probs.append(prob)

    probs = np.array(probs)
    
    return leaf_nodes, probs


def log_binomial_model(S: int, u: float, v: float, p: float, t: int):
    """
    S: strike price
    u: factor of increase
    v: factor of decrease
    p: probability of increase
    """
    leaf_nodes = np.array([0]*(t+1))
    probs = []

    # 0, 1, 2, 3, 4
    for h in np.arange(t+1):
        val = (u**h * v**(t-h)) 
        leaf_nodes[h] = val * S
        prob = -np.log(comb(t, h)) - np.log(p**h) - np.log((1-p)**(t-h))
        probs.append(prob)

    probs = np.array(probs)
    
    return leaf_nodes, probs


def main():
    cli_args = vars(cli_parser())
    print(f"FILE: {cli_args}")

    msft_df = pd.read_csv("../data/msft_per_day.csv")

    N = msft_df.shape[0]
    tomorrow = msft_df["close"][1:N].values
    today = msft_df["close"][0:N-1].values
    returns = np.divide(np.subtract(tomorrow, today), today)
    S = tomorrow[-1]

    timestep = 1/252
    mean, std = returns.mean(), returns.std()
    drift = mean / timestep
    volatility = std / sqrt(timestep) # std = volatility * sqrt(timestep)
    u = 1 + std
    v = 1 - std
    p = .5 + drift*sqrt(timestep)/(2*volatility)
    t = 75

    # print(f"u: {u} -- v: {v} -- p: {p} -- std: {std} -- vol: {volatility} -- drift: {drift} -- mean: {mean}")

    leaf_nodes, probs = binomial_model(S, u, v, p, t)
    logleaf_nodes, log_probs = log_binomial_model(S, u, v, p, t)

    # print(f"Possible Final Expiry Prices (Start: {S}): {leaf_nodes}")
    # print(f"Probailities: {probs}\n\nLog Probabilities: {log_probs}")
    # print(f"Probabilities Normalize: {log_probs/log_probs.sum()}")


if __name__ == "__main__":
    main()