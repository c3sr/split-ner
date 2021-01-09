import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple

import dataclasses
import numpy as np
import torch
import wandb
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass
from transformers.training_args import default_logdir


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

    def to_tsv_form(self):
        return "\t".join([self.text, self.pos_tag, self.dep_tag, self.tag])


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

    def to_tsv_form(self):
        return "\n".join([token.to_tsv_form() for token in self.tokens])


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)


def set_wandb(wandb_dir):
    os.environ["WANDB_WATCH"] = "all"
    wandb.init(project=os.getenv("WANDB_PROJECT", "sec-ner"), dir=wandb_dir)


def parse_config(parser: HfArgumentParser, json_file: str) -> Tuple[DataClass, ...]:
    data = json.loads(Path(json_file).read_text())

    curr_run_output_dir = os.path.join(data["out_root"], data["dataset_dir"], data["model_name"])
    data["output_dir"] = os.path.join(curr_run_output_dir, "checkpoints")
    data["logging_dir"] = os.path.join(curr_run_output_dir, default_logdir())

    outputs = []
    for dtype in parser.dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in data.items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return (*outputs,)
