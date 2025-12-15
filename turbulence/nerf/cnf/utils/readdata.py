import numpy as np
import glob
import re
def load_elbow_flow(path):
    return np.load(f"{path}")[1:]

def load_channel_flow(
    path,
    t_start=0,
    t_end=1200,
    t_every=1,
):
    return np.load(f"{path}")[t_start:t_end:t_every]

def load_periodic_hill_flow(path):
    data = np.load(f"{path}")
    return data

def load_3d_flow(path):
    data = np.load(f"{path}")
    return data






