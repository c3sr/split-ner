import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from shutil import copyfile
from typing import Tuple

import dataclasses
import numpy as np
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass
from transformers.training_args import default_logdir

logger = logging.getLogger(__name__)


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
    import torch
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
    import wandb
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


def read_data(data_path, sep="\t"):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        sent = []
        for line in f:
            line = line.strip()
            if line:
                sent.append(line.split(sep))
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


def make_shorter_dataset(inp_path, shrink_factor=0.1, seed=42):
    out_path = "{0}_{1}_sd{2}".format(inp_path, int(shrink_factor * 100), seed)
    os.makedirs(out_path, exist_ok=True)
    set_all_seeds(seed)
    make_shorter_dataset_util(os.path.join(inp_path, "train.tsv"), os.path.join(out_path, "train.tsv"), shrink_factor)
    make_shorter_dataset_util(os.path.join(inp_path, "dev.tsv"), os.path.join(out_path, "dev.tsv"), shrink_factor)
    # copyfile(os.path.join(inp_path, "dev.tsv"), os.path.join(out_path, "dev.tsv"))
    copyfile(os.path.join(inp_path, "test.tsv"), os.path.join(out_path, "test.tsv"))
    copyfile(os.path.join(inp_path, "tag_vocab.txt"), os.path.join(out_path, "tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "tag_names.txt"), os.path.join(out_path, "tag_names.txt"))
    copyfile(os.path.join(inp_path, "pos_tag_vocab.txt"), os.path.join(out_path, "pos_tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "dep_tag_vocab.txt"), os.path.join(out_path, "dep_tag_vocab.txt"))


def make_shorter_dataset_util(inp_data_path, out_data_path, shrink_factor):
    data = read_data(inp_data_path)
    n = len(data)
    vec = np.random.choice(np.arange(n), int(shrink_factor * n), replace=False)
    new_data = [data[k] for k in vec]
    write_data(new_data, out_data_path)


def make_k_shot_dataset(inp_path, k, seed):
    setup_logging()
    set_all_seeds(seed)
    out_path = "{0}_{1}shot_sd{2}".format(inp_path, k, seed)
    os.makedirs(out_path, exist_ok=True)
    make_k_shot_dataset_util(os.path.join(inp_path, "train.tsv"), os.path.join(out_path, "train.tsv"), k)
    make_k_shot_dataset_util(os.path.join(inp_path, "dev.tsv"), os.path.join(out_path, "dev.tsv"), k)
    copyfile(os.path.join(inp_path, "test.tsv"), os.path.join(out_path, "test.tsv"))
    copyfile(os.path.join(inp_path, "tag_vocab.txt"), os.path.join(out_path, "tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "tag_names.txt"), os.path.join(out_path, "tag_names.txt"))
    copyfile(os.path.join(inp_path, "pos_tag_vocab.txt"), os.path.join(out_path, "pos_tag_vocab.txt"))
    copyfile(os.path.join(inp_path, "dep_tag_vocab.txt"), os.path.join(out_path, "dep_tag_vocab.txt"))


def make_k_shot_dataset_util(inp_data_path, out_data_path, k):
    data = read_data(inp_data_path)
    tag_to_sent_map_tmp = defaultdict(set)
    for sent_index, sent in enumerate(data):
        for token in sent:
            if token[-1][0] in ["B", "S"]:
                tag_to_sent_map_tmp[token[-1][2:]].add(sent_index)
    tag_to_sent_map = {t: sorted(list(tag_to_sent_map_tmp[t])) for t in tag_to_sent_map_tmp}
    sorted_tags = sorted(list(tag_to_sent_map.keys()), key=lambda t: (len(tag_to_sent_map[t]), t))
    tag_cnt = {t: k for t in sorted_tags}
    tag_index = 0
    k_shot_data = []
    while tag_index < len(sorted_tags):
        tag = sorted_tags[tag_index]
        if tag_cnt[tag] <= 0:
            tag_index += 1
            continue
        sent_indices = tag_to_sent_map[tag]
        if len(sent_indices) == 0:
            logger.warning("{0} has less than {1} samples in {2}".format(tag, k, inp_data_path))
            tag_index += 1
            continue
        index = np.random.choice(sent_indices)
        sent = data[index]
        k_shot_data.append(sent)
        for token in sent:
            if token[-1][0] in ["B", "S"]:
                t = token[-1][2:]
                if index in tag_to_sent_map[t]:
                    tag_to_sent_map[t].remove(index)
                tag_cnt[t] -= 1
    write_data(k_shot_data, out_data_path)


def read_mit_data(file_path):
    raw_data = read_data(file_path)
    new_data = []
    for sent in raw_data:
        new_sent = []
        for offset, tup in enumerate(sent):
            new_sent.append(Token(text=tup[1], tags=[tup[0]], offset=offset))
        new_data.append(new_sent)
    return new_data


def add_pos_dep_features(data, spacy_model_name, add_pos=True, add_dep=True):
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(spacy_model_name)
    tokenizer_map = dict()
    nlp.tokenizer = lambda x: Doc(nlp.vocab, tokenizer_map[x])
    print("total: {0}".format(len(data)))
    for index, sent in enumerate(data):
        if index % 1000 == 0:
            print("processing sentence: {0}".format(index))
        words = [tok.text for tok in sent]
        sent_text = " ".join(words)
        tokenizer_map[sent_text] = words
        doc = nlp(sent_text)
        for i, token in enumerate(doc):
            if add_pos:
                sent[i].pos_tag = token.tag_
            if add_dep:
                sent[i].dep_tag = token.dep_
    return data


def write_token_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for sent in data:
            for tok in sent:
                f.write("{0}\t{1}\t{2}\t{3}\n".format(tok.text, tok.pos_tag, tok.dep_tag, tok.tags[0]))
            f.write("\n")


def partition_data(data, ratio):
    set_all_seeds(42)
    n = len(data)
    vec = np.random.choice(np.arange(n), int(ratio * n), replace=False)
    part1 = [data[k] for k in vec]
    part2 = [data[k] for k in range(n) if k not in vec]
    return part1, part2


def process_mit_corpora(dir_path, raw_file_prefix):
    raw_root_dir = os.path.join(dir_path, "raw")
    orig_train = read_mit_data(os.path.join(raw_root_dir, "{0}train.bio".format(raw_file_prefix)))
    orig_train = add_pos_dep_features(orig_train, "en_core_web_sm")
    test = read_mit_data(os.path.join(raw_root_dir, "{0}test.bio".format(raw_file_prefix)))
    test = add_pos_dep_features(test, "en_core_web_sm")
    train, dev = partition_data(orig_train, 0.9)
    generate_dataset_files(train, dev, test, dir_path)


def generate_dataset_files(train, dev, test, dir_path):
    corpus = train + dev + test
    tag_vocab = sorted(list(set([tok.tags[0] for sent in corpus for tok in sent])))
    pos_tag_vocab = sorted(list(set([tok.pos_tag for sent in corpus for tok in sent])))
    dep_tag_vocab = sorted(list(set([tok.dep_tag for sent in corpus for tok in sent])))
    write_token_data(train, os.path.join(dir_path, "train.tsv"))
    write_token_data(dev, os.path.join(dir_path, "dev.tsv"))
    write_token_data(test, os.path.join(dir_path, "test.tsv"))
    with open(os.path.join(dir_path, "tag_vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tag_vocab) + "\n")
    with open(os.path.join(dir_path, "pos_tag_vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pos_tag_vocab) + "\n")
    with open(os.path.join(dir_path, "dep_tag_vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(dep_tag_vocab) + "\n")
    with open(os.path.join(dir_path, "tag_names.txt"), "w", encoding="utf-8") as f:
        for tag in tag_vocab:
            f.write("{0}\t{1}\n".format(tag, tag))


def read_wnut_data(file_path):
    raw_data = read_data(file_path)
    new_data = []
    for sent in raw_data:
        new_sent = []
        for offset, tup in enumerate(sent):
            new_sent.append(Token(text=tup[0], tags=[tup[1]], offset=offset))
        new_data.append(new_sent)
    return new_data


def process_wnut_corpus(corpus_path):
    raw_root_dir = os.path.join(corpus_path, "raw")
    train = add_pos_dep_features(read_wnut_data(os.path.join(raw_root_dir, "train.txt")), "en_core_web_sm")
    dev = add_pos_dep_features(read_wnut_data(os.path.join(raw_root_dir, "dev.txt")), "en_core_web_sm")
    test = add_pos_dep_features(read_wnut_data(os.path.join(raw_root_dir, "test.txt")), "en_core_web_sm")
    generate_dataset_files(train, dev, test, corpus_path)


def read_atis_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tok_sent, tag_sent = line.split("\t")
            tokens = tok_sent.split()[1:-1]
            tags = tag_sent.split()[1:-1]
            sent = []
            for i in range(len(tokens)):
                sent.append(Token(text=tokens[i], tags=[tags[i]], offset=i))
            data.append(sent)
    return data


def process_atis_corpus(corpus_path):
    raw_root_dir = os.path.join(corpus_path, "raw")
    train = add_pos_dep_features(read_atis_data(os.path.join(raw_root_dir, "atis.train.iob.txt")), "en_core_web_sm")
    dev = add_pos_dep_features(read_atis_data(os.path.join(raw_root_dir, "atis.dev.iob.txt")), "en_core_web_sm")
    test = add_pos_dep_features(read_atis_data(os.path.join(raw_root_dir, "atis.test.iob.txt")), "en_core_web_sm")
    generate_dataset_files(train, dev, test, corpus_path)


def read_onto_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        sent = []
        num_parts = 0
        for line in f:
            line = line.strip()
            s = line.split()
            if line.startswith("#end document"):
                continue
            if line.startswith("#begin document"):
                num_parts += 1
                continue

            if not line or len(s) < 11:
                if len(sent) > 0:
                    data.append(sent)
                    sent = []
                continue
            text = s[3]
            pos_tag = s[4]
            tag = s[10]
            token = Token(text=text, tags=[tag], offset=len(sent), pos_tag=pos_tag)
            sent.append(token)
        if len(sent) > 0:
            data.append(sent)
    return data, num_parts


def process_onto_entity_spans(sent):
    spans = []
    for index, tok in enumerate(sent):
        if tok.tags[0].startswith("("):
            tag = tok.tags[0][1:-1]
            spans.append((tag, PairSpan(index, index)))
        if tok.tags[0].endswith(")"):
            spans[-1][1].end = index
    for tok in sent:
        tok.tags[0] = "O"
    for span in spans:
        sent[span[1].start].tags[0] = "B-{0}".format(span[0])
        for i in range(span[1].start + 1, span[1].end + 1):
            sent[i].tags[0] = "I-{0}".format(span[0])
    return sent


def process_onto_corpus_split(root_path, split_type):
    path = os.path.join(root_path, split_type, "data", "english", "annotations")
    data = []
    num_docs = 0
    num_parts = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file_path)
            version, quality, layer = ext[1:].split("_")
            if not (layer == "conll" and quality == "gold"):
                continue
            doc_data, doc_num_parts = read_onto_data(file_path)
            num_parts += doc_num_parts
            num_docs += 1
            data.extend(doc_data)

    for i in range(len(data)):
        data[i] = process_onto_entity_spans(data[i])

    print("#docs: {0} | #parts: {1}".format(num_docs, num_parts))
    return data


def count_onto_entities(data):
    num_entities = 0
    for sent in data:
        sent_num_entities = 0
        for tok in sent:
            if tok.tags[0][0] == "B":
                sent_num_entities += 1
        num_entities += sent_num_entities
    print("#entities: {0}".format(num_entities))


def process_onto_corpus(corpus_path):
    path = os.path.join(corpus_path, "raw", "conll-2012", "v4", "data")
    train = process_onto_corpus_split(path, "train")
    dev = process_onto_corpus_split(path, "development")
    test = process_onto_corpus_split(path, "test")

    train = add_pos_dep_features(train, "en_core_web_sm", add_pos=False, add_dep=True)
    dev = add_pos_dep_features(dev, "en_core_web_sm", add_pos=False, add_dep=True)
    test = add_pos_dep_features(test, "en_core_web_sm", add_pos=False, add_dep=True)

    write_token_data(train, os.path.join(corpus_path, "train.tsv"))
    write_token_data(dev, os.path.join(corpus_path, "dev.tsv"))
    write_token_data(test, os.path.join(corpus_path, "test.tsv"))
    generate_dataset_files(train, dev, test, corpus_path)


def read_conllpp_data(file_path):
    raw_data = read_data(file_path, sep=" ")
    new_data = []
    for sent in raw_data:
        new_sent = []
        for offset, tup in enumerate(sent):
            pos_tag = "O" if tup[1] == "-X-" else tup[1]
            new_sent.append(Token(text=tup[0], tags=[tup[3]], pos_tag=pos_tag, offset=offset))
        new_data.append(new_sent)
    return new_data


def process_conllpp_corpus(corpus_path):
    raw_path = os.path.join(corpus_path, "raw")
    train = read_conllpp_data(os.path.join(raw_path, "conllpp_train.txt"))
    dev = read_conllpp_data(os.path.join(raw_path, "conllpp_dev.txt"))
    test = read_conllpp_data(os.path.join(raw_path, "conllpp_test.txt"))

    train = add_pos_dep_features(train, "en_core_web_sm", add_pos=False, add_dep=True)
    dev = add_pos_dep_features(dev, "en_core_web_sm", add_pos=False, add_dep=True)
    test = add_pos_dep_features(test, "en_core_web_sm", add_pos=False, add_dep=True)

    write_token_data(train, os.path.join(corpus_path, "train.tsv"))
    write_token_data(dev, os.path.join(corpus_path, "dev.tsv"))
    write_token_data(test, os.path.join(corpus_path, "test.tsv"))
    generate_dataset_files(train, dev, test, corpus_path)


def read_onto_final_data(words_path, labels_path):
    with open(words_path, "r", encoding="utf-8") as f:
        text = [line.split() for line in f]
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.split() for line in f]
    data = []
    for sent_index in range(len(text)):
        sent_text = text[sent_index]
        sent_labels = labels[sent_index]
        assert len(sent_text) == len(sent_labels), "Error parsing sent: {0} (text: {1}, labels: {2})" \
            .format(sent_index, len(sent_text), len(sent_labels))
        sent = []
        for word_index in range(len(sent_text)):
            sent.append(Token(text=sent_text[word_index], tags=[sent_labels[word_index]]))
        data.append(sent)
    return data


def process_onto_final_corpus(corpus_path):
    raw_path = os.path.join(corpus_path, "raw_committed")
    train = read_onto_final_data(os.path.join(raw_path, "train.words"), os.path.join(raw_path, "train.ner"))
    dev = read_onto_final_data(os.path.join(raw_path, "dev.words"), os.path.join(raw_path, "dev.ner"))
    test = read_onto_final_data(os.path.join(raw_path, "test.words"), os.path.join(raw_path, "test.ner"))

    train = add_pos_dep_features(train, "en_core_web_sm", add_pos=True, add_dep=True)
    dev = add_pos_dep_features(dev, "en_core_web_sm", add_pos=True, add_dep=True)
    test = add_pos_dep_features(test, "en_core_web_sm", add_pos=True, add_dep=True)

    write_token_data(train, os.path.join(corpus_path, "train.tsv"))
    write_token_data(dev, os.path.join(corpus_path, "dev.tsv"))
    write_token_data(test, os.path.join(corpus_path, "test.tsv"))
    generate_dataset_files(train, dev, test, corpus_path)

# def process_i2b2_corpus(corpus_path):
#     raw_path = os.path.join(corpus_path, "raw")
#     orig_train = read_i2b2_data(os.path.join(raw_path, "train"))
#     test = read_i2b2_data(os.path.join(raw_path, "test"))
#
#     train = add_pos_dep_features(train, "en_core_sci_sm", add_pos=False, add_dep=True)
#     dev = add_pos_dep_features(dev, "en_core_sci_sm", add_pos=False, add_dep=True)
#     test = add_pos_dep_features(test, "en_core_sci_sm", add_pos=False, add_dep=True)
#
#     write_token_data(train, os.path.join(raw_path, "train.tsv"))
#     write_token_data(dev, os.path.join(raw_path, "dev.tsv"))
#     write_token_data(test, os.path.join(raw_path, "test.tsv"))
#     generate_dataset_files(train, dev, test, corpus_path)
