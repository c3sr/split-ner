import json
import os
import random

import numpy as np
import torch


class Token:
    def __init__(self, text, tag, offset=None, pos_tag=None, dep_tag=None, guidance_tag=None):
        self.offset = offset
        self.text = text
        self.tag = tag
        self.pos_tag = pos_tag
        self.dep_tag = dep_tag
        self.guidance_tag = guidance_tag

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.text, self.tag, self.offset, self.pos_tag, self.dep_tag,
                                                       self.guidance_tag)

    def __repr__(self):
        return self.__str__()


class BertToken:
    def __init__(self, bert_id, token):
        self.bert_id = bert_id
        self.token = token

    def __str__(self):
        return "({0}, {1})".format(self.bert_id, self.token)

    def __repr__(self):
        return self.__str__()


class Sentence:
    def __init__(self, tokens=None, bert_tokens=None):
        self.tokens = tokens
        self.bert_tokens = bert_tokens


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def parse_config(config_file):
    with open(config_file, "r") as f:
        d = json.load(f)
    return json.loads(json.dumps(d), object_hook=Config)


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_compute_device(is_cpu_forced):
    if is_cpu_forced or not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def set_absolute_paths(config):
    config.data.tags_path = os.path.join(config.data.data_dir, config.data.tags_path)
    config.data.train_path = os.path.join(config.data.data_dir, config.data.train_path)
    config.data.dev_path = os.path.join(config.data.data_dir, config.data.dev_path)
    config.data.test_path = os.path.join(config.data.data_dir, config.data.test_path)
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, config.name)
    config.data.out_dir = os.path.join(config.data.out_dir, config.name)
