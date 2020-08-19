import os
import random

import numpy as np
import torch


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_sum_exp(vec):
    max_score, _ = torch.max(vec, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(vec)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), -1))


def get_compute_device(is_cpu_forced):
    if is_cpu_forced or not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def parse_emb_file(emb_path, has_header_line=False):
    emb_dict = dict()
    with open(emb_path, "r") as f:
        for index, line in enumerate(f):
            if has_header_line and index == 0:
                continue
            line = line.strip()
            if line:
                s = line.split(" ")
                key = s[0]
                # handling special cases
                if key == "(AND_G#cell_type_G#cell_type)":
                    key = "(AND G#cell_type G#cell_type)"
                elif key == "(AND_G#protein_complex_G#protein_complex)":
                    key = "(AND G#protein_complex G#protein_complex)"
                emb_dict[key] = [float(x) for x in s[1:]]
    return emb_dict
