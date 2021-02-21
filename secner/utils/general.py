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
    def __init__(self, bert_id, sub_text, token_type, token, is_head):
        self.bert_id = bert_id
        self.sub_text = sub_text
        self.token_type = token_type
        self.token = token
        self.is_head = is_head

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4})".format(self.bert_id, self.sub_text, self.token_type, self.token,
                                                  self.is_head)

    def __repr__(self):
        return self.__str__()


class Sentence:
    def __init__(self, tokens=None, bert_tokens=None):
        self.tokens = tokens
        self.bert_tokens = bert_tokens

    def to_tsv_form(self):
        return "\n".join([token.to_tsv_form() for token in self.tokens])


class PairSpan:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return "({0}, {1})".format(self.start, self.end)

    def __repr__(self):
        return self.__str__()


class Context:
    def __init__(self, sentence=None, entity=None, entity_text=None, bert_tokens=None):
        self.sentence = sentence
        self.entity = entity
        self.entity_text = entity_text
        self.bert_tokens = bert_tokens


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
    os.makedirs(os.path.join(wandb_dir, "wandb"), exist_ok=True)
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


def generate_tag_names_for_underscore_separated_tags(dataset_name):
    tags = set()
    prefix = "../../data/{0}".format(dataset_name)
    with open("{0}/tag_vocab.txt".format(prefix), "r") as f:
        for line in f:
            line = line.strip()
            if line[2:]:
                tags.add(line[2:])

    tags = sorted(list(tags))
    with open("{0}/tag_names.txt".format(prefix), "w", encoding="utf-8") as f:
        for tag in tags:
            f.write("{0}\t{1}\n".format(tag, " ".join(tag.split("_"))))


# used for all parsed datasets (Eg. BioNLP13CG, OntoNotes, CoNLL, JNLPBA etc.)
def generate_aux_tag_vocab_from_data(dataset_name):
    pos_vocab = set()
    dep_vocab = set()
    _parse_data_and_add_to_vocab(dataset_name, "train", dep_vocab, pos_vocab)
    _parse_data_and_add_to_vocab(dataset_name, "dev", dep_vocab, pos_vocab)
    _parse_data_and_add_to_vocab(dataset_name, "test", dep_vocab, pos_vocab)

    pos_vocab = sorted(list(pos_vocab))
    dep_vocab = sorted(list(dep_vocab))

    with open("../../data/{0}/pos_tag_vocab.txt".format(dataset_name), "w", encoding="utf-8") as f:
        for w in pos_vocab:
            f.write("{0}\n".format(w))

    with open("../../data/{0}/dep_tag_vocab.txt".format(dataset_name), "w", encoding="utf-8") as f:
        for w in dep_vocab:
            f.write("{0}\n".format(w))


def _parse_data_and_add_to_vocab(dataset_name, corpus_type, dep_vocab, pos_vocab):
    with open("../../data/{0}/{1}.tsv".format(dataset_name, corpus_type), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s = line.split("\t")
                pos_vocab.add(s[1])
                dep_vocab.add(s[2])
