import json
import logging
import os
import random
from pathlib import Path
from shutil import copyfile
from typing import Tuple

import dataclasses
import numpy as np
import torch
import wandb
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass
from transformers.training_args import default_logdir


class Token:
    def __init__(self, text, tags, offset=None, pos_tag=None, dep_tag=None, guidance_tag=None):
        self.offset = offset
        self.text = text
        self.tags = tags
        self.pos_tag = pos_tag
        self.dep_tag = dep_tag
        self.guidance_tag = guidance_tag

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.text, self.tags, self.offset, self.pos_tag, self.dep_tag,
                                                       self.guidance_tag)

    def __repr__(self):
        return self.__str__()

    def to_tsv_form(self):
        return "\t".join([self.text, self.pos_tag, self.dep_tag] + self.tags)


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

    def __str__(self):
        return self.tokens.__str__()

    def __repr__(self):
        return self.__str__()


class PairSpan:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return "({0}, {1})".format(self.start, self.end)

    def __repr__(self):
        return self.__str__()


class Context:
    def __init__(self, sentence=None, entity=None, entity_text=None, bert_tokens=None, mention_span: PairSpan = None):
        self.sentence = sentence
        self.entity = entity
        self.entity_text = entity_text
        self.bert_tokens = bert_tokens
        self.mention_span = mention_span


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
    with open("{0}/tag_vocab.txt".format(prefix), "r", encoding="utf-8") as f:
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


def remove_punct_rows_from_dataset(data_path):
    data = read_data(data_path)
    for sent in data:
        sent[:] = [line for line in sent if line[0] not in list("-()/")]
    write_data(data, data_path)


def read_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        sent = []
        for line in f:
            line = line.strip()
            if line:
                sent.append(line.split("\t"))
            else:
                data.append(sent)
                sent = []
        if len(sent) > 0:
            data.append(sent)
    return data


def write_data(data, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        for sent in data:
            for line in sent:
                f.write("\t".join(line) + "\n")
            f.write("\n")


def make_shorter_dataset(inp_path, shrink_factor=0.1):
    out_path = "{0}_{1}".format(inp_path, int(shrink_factor * 100))
    os.makedirs(out_path, exist_ok=True)
    make_shorter_dataset_util(os.path.join(inp_path, "train.tsv"), os.path.join(out_path, "train.tsv"), shrink_factor)
    make_shorter_dataset_util(os.path.join(inp_path, "dev.tsv"), os.path.join(out_path, "dev.tsv"), shrink_factor)
    # copyfile(os.path.join(inp_path, "dev.tsv"), os.path.join(out_path, "dev.tsv"))
    copyfile(os.path.join(inp_path, "test.tsv"), os.path.join(out_path, "test.tsv"))
    copyfile(os.path.join(inp_path, "tag_vocab.txt"), os.path.join(out_path, "tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "tag_names.txt"), os.path.join(out_path, "tag_names.txt"))
    copyfile(os.path.join(inp_path, "pos_tag_vocab.txt"), os.path.join(out_path, "pos_tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "dep_tag_vocab.txt"), os.path.join(out_path, "dep_tag_vocab.txt"))


def make_shorter_dataset_util(inp_data_path, out_data_path, shrink_factor):
    set_all_seeds(42)
    data = read_data(inp_data_path)
    n = len(data)
    vec = np.random.choice(np.arange(n), int(shrink_factor * n), replace=False)
    new_data = [data[k] for k in vec]
    write_data(new_data, out_data_path)
