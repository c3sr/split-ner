import argparse
import logging
import re
from collections import defaultdict

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer

from splitner.additional_args import AdditionalArguments
from splitner.utils.general import Token, set_all_seeds, BertToken, Sentence, parse_config, setup_logging, PairSpan

# logging.basicConfig(filename='dataset.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class NerDataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerDataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()

        if self.args.detect_spans:
            self.tag_vocab = self.get_span_tag_vocab()
        else:
            self.tag_vocab = NerDataset.parse_tag_vocab(self.args.tag_vocab_path)
            self.add_tags_as_per_tagging_scheme()

        self.pos_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.pos_tag_vocab_path, self.args.none_tag,
                                                            self.args.use_pos_tag)
        self.dep_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.dep_tag_vocab_path, self.args.none_tag,
                                                            self.args.use_dep_tag)

        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_first_sep_token, self.bert_second_sep_token = \
            NerDataset.get_bert_special_tokens(self.tokenizer, self.args.none_tag)
        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)
        self.filter_tags()
        self.split_tags()
        self.parse_dataset()

    def get_span_tag_vocab(self):
        tag_vocab = ["B-ENTITY", "I-ENTITY"]
        if self.args.tagging == "bioe":
            tag_vocab.append("E-ENTITY")
        if self.args.tagging == "bioes":
            tag_vocab.append("E-ENTITY")
            tag_vocab.append("S-ENTITY")
        tag_vocab.append("O")
        return tag_vocab

    def add_tags_as_per_tagging_scheme(self):
        e_tags = ["E-" + tag[2:] for tag in self.tag_vocab if tag.startswith("B-")]
        s_tags = ["S-" + tag[2:] for tag in self.tag_vocab if tag.startswith("B-")]
        # Note: in "bioe" and "bioes" schemes, the "O" tag comes in the middle in the tag_vocab
        if self.args.tagging == "bioe":
            self.tag_vocab += e_tags
        if self.args.tagging == "bioes":
            self.tag_vocab += e_tags + s_tags

    def set_corpus_path(self):
        if self.corpus_type == "train":
            return self.args.train_path
        if self.corpus_type == "dev":
            return self.args.dev_path
        if self.corpus_type == "test":
            return self.args.test_path
        return None

    @staticmethod
    def parse_tag_vocab(vocab_path):
        tag_vocab = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tag_vocab.append(line)
        return tag_vocab

    @staticmethod
    def parse_aux_tag_vocab(vocab_path, none_tag, do_task=True):
        if not do_task:
            return []
        vocab = [none_tag]
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    vocab.append(line)
        return vocab

    def split_tags(self):
        if not self.args.split_tags:
            return
        if self.args.dataset_dir == "bio":
            self.tag_vocab.append("B-Symbolic_simple_chemical")
            self.tag_vocab.append("I-Symbolic_simple_chemical")
            for sent in self.sentences:
                spans = NerDataset.get_spans(sent)
                for sp in spans["Simple_chemical"]:
                    mention = " ".join([sent.tokens[i].text for i in range(sp.start, sp.end + 1)])
                    if re.search(r"[^A-Za-z0-9]|\d", mention):
                        k = sent.tokens[sp.start].tags.index("B-Simple_chemical")
                        sent.tokens[sp.start].tags[k] = "B-Symbolic_simple_chemical"
                        for index in range(sp.start + 1, sp.end + 1):
                            k = sent.tokens[index].tags.index("I-Simple_chemical")
                            sent.tokens[index].tags[k] = "I-Symbolic_simple_chemical"

    def filter_tags(self):
        if self.args.filter_tags is None:
            return
        permissible_tags = [self.args.none_tag]
        permissible_tags.extend(self.args.filter_tags)
        permissible_tags = set(permissible_tags)
        new_tag_vocab = []
        for tag in self.tag_vocab:
            if tag == self.args.none_tag or tag[2:] in permissible_tags:
                new_tag_vocab.append(tag)
        self.tag_vocab = new_tag_vocab

        for sent in self.sentences:
            for tok in sent.tokens:
                new_tags = tok.tags
                for tag in tok.tags:
                    if tag[2:] not in permissible_tags:
                        new_tags.remove(tag)
                if len(new_tags) == 0:
                    new_tags.append(self.args.none_tag)
                tok.tags = new_tags

    @staticmethod
    def get_spans(sentence):
        spans = defaultdict(list)
        for index, tok in enumerate(sentence.tokens):
            for tag in tok.tags:
                if tag.startswith("B-"):
                    spans[tag[2:]].append(PairSpan(index, index))
                elif tag.startswith("I-"):
                    spans[tag[2:]][-1].end = index
        return spans

    def parse_dataset(self):
        for index in range(len(self.sentences)):
            self.process_sentence(index)

    @staticmethod
    def read_dataset(file_path, args: AdditionalArguments):
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            tokens = []
            offset = 0
            for line in f:
                line = line.strip()
                if line:
                    row = line.split("\t")
                    for rt in NerDataset.get_row_tokens(row, args):
                        tokens.append(Token(text=rt.text, pos_tag=rt.pos_tag, dep_tag=rt.dep_tag, tags=rt.tags,
                                            offset=offset))
                        offset += 1
                else:
                    sentences.append(Sentence(tokens))
                    tokens = []
                    offset = 0
        if len(tokens) > 0:
            sentences.append(Sentence(tokens))
        if args.debug_mode:
            sentences = sentences[:10]
        return sentences

    @staticmethod
    def get_row_tokens(row, args: AdditionalArguments):
        text = row[0]
        pos_tag = row[1] if args.data_pos_dep else None
        dep_tag = row[2] if args.data_pos_dep else None
        tags = row[3:] if args.data_pos_dep else row[1:]
        if tags == [args.none_tag] or args.use_pattern == "none":
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tags=tags)]
        pattern_text = NerDataset.make_pattern_type0(text)
        if args.use_pattern == "only":
            return [Token(text=pattern_text, pos_tag=pos_tag, dep_tag=dep_tag, tags=tags)]
        if args.use_pattern == "both":
            # E.g.: ABC, a uuu, is investing ...
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tags=tags),
                    Token(text=",", pos_tag=",", dep_tag="punct", tags=[args.none_tag]),
                    Token(text="a", pos_tag="DT", dep_tag="det", tags=[args.none_tag]),
                    Token(text=pattern_text, pos_tag="NN", dep_tag="appos", tags=[args.none_tag]),
                    Token(text=",", pos_tag=",", dep_tag="punct", tags=[args.none_tag])]

    @staticmethod
    def make_pattern(text, pattern_type):
        if pattern_type == "0":
            return NerDataset.make_pattern_type0(text)
        if pattern_type == "1":
            return NerDataset.make_pattern_type1(text)
        if pattern_type == "2":
            return NerDataset.make_pattern_type2(text)
        if pattern_type == "3":
            return NerDataset.make_pattern_type3(text)
        if pattern_type == "4":
            return NerDataset.make_pattern_type4(text)
        raise NotImplementedError

    @staticmethod
    def make_pattern_type0(text):
        pattern_text = ""
        for c in text:
            if "a" <= c <= "z":
                pattern_text += "l"
            elif "A" <= c <= "Z":
                pattern_text += "u"
            else:
                pattern_text += c
        return pattern_text

    @staticmethod
    def make_pattern_type1(text):
        if text == "[CLS]":
            return "C"
        if text == "[SEP]":
            return "S"
        if re.fullmatch(r"[a-z]+", text):
            return "L"
        if re.fullmatch(r"[A-Z]+", text):
            return "U"
        if re.fullmatch(r"[A-Z][a-z]+", text):
            return "F"
        if re.fullmatch(r"[A-Za-z]+", text):
            return "M"
        # for tokens with digits/punctuations
        return NerDataset.make_pattern_type0(text)

    @staticmethod
    #YJP: added CLS and SEP
    def make_pattern_type2(text):
        if text == "[CLS]":
            return "C"
        if text == "[SEP]":
            return "S"

        pattern_text = ""
        for c in text:
            if "a" <= c <= "z":
                pattern_text += "l"
            elif "A" <= c <= "Z":
                pattern_text += "u"
            elif "0" <= c <= "9":
                pattern_text += "d"
            else:
                pattern_text += c

        return pattern_text

    @staticmethod
    def make_pattern_type3(text):
        if text == "[CLS]":
            return "C"
        if text == "[SEP]":
            return "S"
        if re.fullmatch(r"[a-z]+", text):
            return "L"
        if re.fullmatch(r"[A-Z]+", text):
            return "U"
        if re.fullmatch(r"[A-Z][a-z]+", text):
            return "F"
        if re.fullmatch(r"[A-Za-z]+", text):
            return "M"
        # for tokens with digits/punctuations
        return NerDataset.make_pattern_type2(text)

    @staticmethod
    def make_pattern_type4(text):
        if text == "[CLS]":
            return "C"
        if text == "[SEP]":
            return "S"

        pattern_text = ""
        pattern = ""
        prev_pattern = ""
        cnt = 0
        for c in text:
            is_symbol = False
            if "a" <= c <= "z":
                pattern = "L"
            elif "A" <= c <= "Z":
                pattern = "U"
            elif "0" <= c <= "9":
                pattern = "D"
            else:
                pattern = c
                is_symbol = True

            if prev_pattern == "":
               prev_pattern = pattern

            if is_symbol:
                pattern_text += pattern
                prev_pattern = pattern
                cnt=0
            elif prev_pattern != pattern:
                pattern_text += prev_pattern + str(cnt)
                prev_pattern = pattern
                cnt=0

            cnt += 1

        if not is_symbol:
            pattern_text += pattern + str(cnt)

        return pattern_text

    @staticmethod
    def get_word_type(text):
        if text == "[CLS]":
            return "C"
        if text == "[SEP]":
            return "S"
        if re.fullmatch(r"[a-z]+", text):
            return "L"
        if re.fullmatch(r"[A-Z]+", text):
            return "U"
        if re.fullmatch(r"[A-Z][a-z]+", text):
            return "F"
        if re.fullmatch(r"[A-Za-z]+", text):
            return "M"
        if re.fullmatch(r"[0-9]+", text):
            return "D"
        if re.fullmatch(r"[^A-Za-z0-9]+", text):
            return "P"
        if re.fullmatch(r"[A-Za-z0-9]+", text):
            return "A"
        return "B"


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        bert_token_ids = [tok.bert_id for tok in sentence.bert_tokens]
        if self.args.model_mode == "roberta_std":
            bert_token_type_ids = [self.tokenizer.pad_token_type_id for _ in sentence.bert_tokens]
        else:
            bert_token_type_ids = [tok.token_type for tok in sentence.bert_tokens]
        bert_head_mask = [tok.is_head for tok in sentence.bert_tokens]
        bert_token_text = [tok.token.text for tok in sentence.bert_tokens]
        bert_sub_token_text = [tok.sub_text for tok in sentence.bert_tokens]
        bert_token_pos = [self.pos_tag_vocab.index(tok.token.pos_tag) for tok in
                          sentence.bert_tokens] if self.args.use_pos_tag else []
        bert_token_dep = [self.dep_tag_vocab.index(tok.token.dep_tag) for tok in
                          sentence.bert_tokens] if self.args.use_dep_tag else []
        # For seq-tagging framework, we use only the first gold tag for each token (not considering nested NER)
        bert_tag_ids = [self.get_tag_index(tok.token.tags[0]) for tok in sentence.bert_tokens]

        return {"input_ids": bert_token_ids,
                "token_type_ids": bert_token_type_ids,
                "head_mask": bert_head_mask,
                "text": bert_token_text,
                "sub_text": bert_sub_token_text,
                "pos_tag": bert_token_pos,
                "dep_tag": bert_token_dep,
                "labels": bert_tag_ids}

    def get_tag_index(self, text_tag):
        if text_tag not in self.tag_vocab:
            text_tag = self.args.none_tag
        return self.tag_vocab.index(text_tag)

    @staticmethod
    def get_bert_special_tokens(tokenizer, none_tag):
        start_text, start_id = tokenizer.cls_token, tokenizer.cls_token_id
        sep_text, sep_id = tokenizer.sep_token, tokenizer.sep_token_id
        start_token = BertToken(bert_id=start_id,
                                sub_text=start_text,
                                token_type=0,
                                token=Token(start_text, [none_tag], offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                                is_head=True)
        first_sep_token = BertToken(bert_id=sep_id,
                                    sub_text=sep_text,
                                    token_type=0,
                                    token=Token(sep_text, [none_tag], offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                                    is_head=True)
        second_sep_token = BertToken(bert_id=sep_id,
                                     sub_text=sep_text,
                                     token_type=1,
                                     token=Token(sep_text, [none_tag], offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                                     is_head=True)
        return start_token, first_sep_token, second_sep_token

    # YJ : updated to support BIOES tagging scheme
    def process_sentence(self, index):
        sentence = self.sentences[index]
        sentence.bert_tokens = [self.bert_start_token]
        for token in sentence.tokens:
            out = self.tokenizer(token.text, add_special_tokens=False, return_offsets_mapping=True)
            token_tag = token.tags[0]

            for i in range(len(out["input_ids"])):
                if token_tag == "O" or token_tag.startswith("I-"):
                    tag = token_tag
                elif i == 0 and (token_tag.startswith("B-") or token_tag.startswith("S-")):
                    tag = token_tag
                elif i == len(out["input_ids"]) and token_tag.startswith("E-"):
                    tag = token_tag
                else:
                    tag = "I-" + token_tag[2:]

                # Handle 'BO' tagging scheme
                # if self.args.tagging == "bo" and tag[:2] == "I-":
                if self.args.tagging == "bo" and (not tag == "O"):
                    tag = "B-" + tag[2:]

                bert_token = Token(token.text, [tag], token.offset, token.pos_tag, token.dep_tag, token.guidance_tag)
                tup = out["offset_mapping"][i]
                sub_text = token.text[tup[0]:tup[1]]
                sentence.bert_tokens.append(BertToken(out["input_ids"][i], sub_text, 0, bert_token, is_head=(i == 0)))

        if self.args.tagging in ["bioe", "bioes"]:
            is_end_token = False
            for i in range(len(sentence.bert_tokens) - 1, -1, -1):
                if sentence.bert_tokens[i].token.tags[0][0] == "I":
                    if is_end_token:
                        if not self.args.use_head_mask or sentence.bert_tokens[i].is_head:
                            tmp = sentence.bert_tokens[i].token.tags[0]
                            sentence.bert_tokens[i].token.tags[0] = "E-" + tmp[2:]
                            is_end_token = False
                else:
                    is_end_token = True

        if self.args.tagging == "bioes":
            if self.args.use_head_mask:
                mention_length = 0
                mention_index = -1
                for i in range(len(sentence.bert_tokens)):
                    if not sentence.bert_tokens[i].is_head:
                        continue
                    if sentence.bert_tokens[i].token.tags[0][0] == "B":
                        if mention_length == 1:
                            tmp = sentence.bert_tokens[mention_index].token.tags[0]
                            sentence.bert_tokens[mention_index].token.tags[0] = "S-" + tmp[2:]
                        mention_length = 1
                        mention_index = i
                    elif sentence.bert_tokens[i].token.tags[0][0] in ["I", "E"]:
                        mention_length += 1
                    elif mention_length == 1:
                        tmp = sentence.bert_tokens[mention_index].token.tags[0]
                        sentence.bert_tokens[mention_index].token.tags[0] = "S-" + tmp[2:]
                        mention_length = 0
                        mention_index = -1
                if mention_length == 1:
                    tmp = sentence.bert_tokens[mention_index].token.tags[0]
                    sentence.bert_tokens[mention_index].token.tags[0] = "S-" + tmp[2:]
            else:
                for i in range(len(sentence.bert_tokens)):
                    if sentence.bert_tokens[i].token.tags[0][0] == "B" and i + 1 < len(sentence.bert_tokens) and \
                            sentence.bert_tokens[i + 1].token.tags[0][0] not in ["I", "E"]:
                        tmp = sentence.bert_tokens[i].token.tags[0]
                        sentence.bert_tokens[i].token.tags[0] = "S-" + tmp[2:]

        if self.args.detect_spans:
            for bert_token in sentence.bert_tokens:
                if bert_token.token.tags[0] != "O":
                    tmp = bert_token.token.tags[0]
                    bert_token.token.tags[0] = tmp[:2] + "ENTITY"

        sentence.bert_tokens = sentence.bert_tokens[:self.args.max_seq_len - 1]
        sentence.bert_tokens.append(self.bert_first_sep_token)

    @staticmethod
    def get_char_ids(batch_text, max_len, vocab):
        max_word_len = max(len(word) for sent in batch_text for word in sent)
        max_word_len = max(max_word_len, 3)  # TODO: Check CNN kernel size issue here
        batch_ids = []
        for sent_text in batch_text:
            sent_ids = []
            for word_text in sent_text:
                word_ids = [(vocab.index(c) + 1) for c in word_text if c in vocab]
                pad_word_len = max_word_len - len(word_ids)
                sent_ids.append(torch.tensor(word_ids + [0] * pad_word_len, dtype=torch.int64))
            pad_len = max_len - len(sent_ids)
            sent_ids += [torch.zeros(max_word_len, dtype=torch.int64)] * pad_len
            batch_ids.append(torch.stack(sent_ids))
        return torch.stack(batch_ids)

    @staticmethod
    def get_pattern_ids(batch_text, max_len, pattern_vocab):
        batch_ids = []
        vocab_size = len(pattern_vocab)
        for sent_text in batch_text:
            sent_ids = []
            for word_text in sent_text:
                if word_text in pattern_vocab:
                    sent_ids.append(pattern_vocab.index(word_text))
                else:
                    sent_ids.append(vocab_size)

            pad_len = max_len - len(sent_ids)
            sent_ids += [0] * pad_len

            batch_ids.append(torch.tensor(sent_ids))

        return torch.stack(batch_ids)

    @staticmethod
    def get_punctuation_vocab_size(punctuation_type):
        if punctuation_type == "type1":
            return 1
        if punctuation_type == "type1-and":
            return 2
        if punctuation_type == "type2":
            return len(list("O.,-/()P"))

    @staticmethod
    def handle_punctuation1(word, punctuation_type):
        all_punctuations = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        if punctuation_type == "type1":
            return 1 if word in all_punctuations else 0
        if punctuation_type == "type1-and":
            if word in all_punctuations:
                return 0
            if word.lower() in ["and"]:
                return 1
            return -1
        if punctuation_type == "type2":
            punctuation_vocab = list(".,-/()")
            if word in punctuation_vocab:
                return punctuation_vocab.index(word)
            if word in all_punctuations:
                # catch all other punctuations (P)
                return len(punctuation_vocab)
            return 0  # non-punctuation (O)
        raise NotImplementedError

    @staticmethod
    def handle_punctuation2(word, punctuation_type):
        all_punctuations = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        if punctuation_type == "type1":
            return 1 if word in all_punctuations else 2
        if punctuation_type == "type1-and":
            if word in all_punctuations:
                return 1
            if word.lower() in ["and"]:
                return 2
            return 3
        if punctuation_type == "type2":
            punctuation_vocab = list(".,-/()")
            if word in punctuation_vocab:
                return punctuation_vocab.index(word) + 1
            if word in all_punctuations:
                # catch all other punctuations (P)
                return len(punctuation_vocab) + 1
            return len(punctuation_vocab) + 2
        raise NotImplementedError

    @staticmethod
    def get_char_vocab():
        # size: 94 (does not include space, newline)
        # additional: can also use list(string.printable) here (size: 100)
        vocab = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        vocab += list("abcdefghijklmnopqrstuvwxyz")
        vocab += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        vocab += list("0123456789")
        return vocab

    @staticmethod
    def get_flair_vocab():
        vocab = NerDataset.get_char_vocab()
        vocab.append(" ")
        return vocab

    @staticmethod
    def get_pattern_vocab(pattern_type):
        vocab = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        if pattern_type == "0":
            vocab += list("ul")
            vocab += list("0123456789")
            return vocab
        if pattern_type == "1":
            vocab += list("ulCSLUFM")
            vocab += list("0123456789")
            return vocab
        if pattern_type == "2":
            vocab += list("CSlud")
            return vocab
        if pattern_type == "3":
            vocab += list("CSLUFMlud")
            return vocab

        if pattern_type == "4":
            vocab += list("CSLUD")
            vocab += list("0123456789")
            return vocab

        raise NotImplementedError

    @staticmethod
    def get_word_type_vocab():
        vocab = list("CSLUFMDPAB")
        return vocab


@dataclass
class NerDataCollator:
    args: AdditionalArguments
    pattern_vocab: list

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model, use_fast=True)

    def __call__(self, features):
        # post-padding
        # YJ changes: 
        # (1) make sure that all the actual value is not 0 as 0 is used for padding 
        # (2) better to group input data by lengths so that all sequences in a batch are in the (almost) same length and then shuffle the groups
        # (3) move to pre-padding as it is preferred? (https://arxiv.org/pdf/1903.07288.pdf, no results for BiLSTM)


        max_len = max(len(entry["labels"]) for entry in features)
        # max_len = self.args.max_seq_len
        batch = dict()

        # input_ids
        # does the BERT's input_id start from 101? 101 is for CLS and 201 is SEP.
        if "input_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["input_ids"])
                entry.append(torch.tensor(features[i]["input_ids"] + [self.tokenizer.pad_token_id] * pad_len))
            batch["input_ids"] = torch.stack(entry)

        # attention_mask
        entry = []
        for i in range(len(features)):
            good_len = len(features[i]["input_ids"])
            pad_len = max_len - good_len
            entry.append(torch.tensor([1] * good_len + [0] * pad_len))
        batch["attention_mask"] = torch.stack(entry)

        # token_type_ids
        if "token_type_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["token_type_ids"])
                entry.append(torch.tensor(features[i]["token_type_ids"] + [self.tokenizer.pad_token_type_id] * pad_len))
            batch["token_type_ids"] = torch.stack(entry)

        # YJ: char_ids and pattern_id add 1 to index and  padd the rest with 0
        # char_ids
        if self.args.use_char_cnn in ["char", "both"]:
            batch_text = [entry[self.args.token_type] for entry in features]
            char_vocab = NerDataset.get_char_vocab()
            batch["char_ids"] = NerDataset.get_char_ids(batch_text, max_len, char_vocab)

        # pattern_ids
        if self.args.use_char_cnn in ["pattern", "both", "both-flair"]:
            batch_pattern = [[NerDataset.make_pattern(word, self.args.pattern_type)
                              for word in entry[self.args.token_type]] for entry in features]
            pattern_vocab = NerDataset.get_pattern_vocab(self.args.pattern_type)
            batch["pattern_ids"] = NerDataset.get_char_ids(batch_pattern, max_len, pattern_vocab)

        # flair_ids
        if self.args.use_char_cnn in ["flair", "both-flair"]:
            flair_vocab = NerDataset.get_flair_vocab()
            start_index, end_index, pad_index = len(flair_vocab), len(flair_vocab) + 1, len(flair_vocab) + 2
            entry_list = []
            entry_boundary = []
            flair_max_len = 0
            for f in features:
                sent_text = f[self.args.token_type]
                sent_ids = [start_index]
                boundary = []
                for word_text in sent_text[:-1]:
                    boundary.append(len(sent_ids) - 1)
                    sent_ids += [flair_vocab.index(c) for c in word_text if c in flair_vocab]
                    sent_ids.append(flair_vocab.index(" "))
                boundary.append(len(sent_ids) - 1)
                sent_ids += [flair_vocab.index(c) for c in sent_text[-1] if c in flair_vocab]
                sent_ids.append(end_index)
                boundary.append(len(sent_ids) - 1)
                pad_len = max_len + 1 - len(boundary)  # count(boundaries) = count(elements) + 1
                entry_boundary.append(torch.tensor(boundary + [-1] * pad_len))
                entry_list.append(sent_ids)
                flair_max_len = max(flair_max_len, len(sent_ids))
            batch["flair_boundary"] = torch.stack(entry_boundary)

            entry = []
            entry_mask = []
            for i in range(len(features)):
                pad_len = flair_max_len - len(entry_list[i])
                entry.append(torch.tensor(entry_list[i] + pad_len * [pad_index]))
                entry_mask.append(torch.tensor([1] * len(entry_list[i]) + [0] * pad_len, dtype=torch.int64))
            batch["flair_ids"] = torch.stack(entry)
            batch["flair_attention_mask"] = torch.stack(entry_mask)

        # YJ: update handle_punctuation not to return 0 or negative value
        if self.args.punctuation_handling != "none" or self.args.loss_type == "ce_punct":
            punct_type = self.args.punctuation_handling
            if self.args.loss_type == "ce_punct":
                # TODO: ce_punct currently does not work with multi-dimensional punctuation_vec (expects 'type1' format)
                punct_type = "type1"
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i][self.args.token_type])
                entry.append(torch.tensor([NerDataset.handle_punctuation2(w, punct_type)
                                           for w in features[i][self.args.token_type]] + [0] * pad_len))
            batch["punctuation_vec"] = torch.stack(entry)

        # YJ: add 1 to index, so padding 0 is fine
        if self.args.word_type_handling != "none":
            entry = []
            word_type_vocab = NerDataset.get_word_type_vocab()
            for i in range(len(features)):
                pad_len = max_len - len(features[i][self.args.token_type])
                # padding tokens get word_type 0, all others get valid word_type indices (1 onwards)
                entry.append(torch.tensor([word_type_vocab.index(NerDataset.get_word_type(w)) + 1
                                           for w in features[i][self.args.token_type]] + [0] * pad_len))
            batch["word_type_ids"] = torch.stack(entry)

        # head_mask
        # YJ: how is this field used?  need to add 1?
        entry = []
        for i in range(len(features)):
            pad_len = max_len - len(features[i]["head_mask"])
            entry.append(torch.tensor(features[i]["head_mask"] + [0] * pad_len))
        batch["head_mask"] = torch.stack(entry)

        # YJ: pos_tag and dep_tag's vocabulary contain [PAD] as the first element, so padding 0 is fine
        # pos_tag
        if self.args.use_pos_tag:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["pos_tag"])
                entry.append(torch.tensor(features[i]["pos_tag"] + [0] * pad_len))
            batch["pos_tag"] = torch.stack(entry)

        # dep_tag
        if self.args.use_dep_tag:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["dep_tag"])
                entry.append(torch.tensor(features[i]["dep_tag"] + [0] * pad_len))
            batch["dep_tag"] = torch.stack(entry)

        # labels
        entry = []
        for i in range(len(features)):
            labels_mod = features[i]["labels"]
            pad_len = max_len - len(labels_mod)
            entry.append(torch.tensor(labels_mod + [-100] * pad_len))
        batch["labels"] = torch.stack(entry)

        if self.args.gold_span_inp == "simple":
            none_index = NerDataset.parse_tag_vocab(self.args.tag_vocab_path).index(self.args.none_tag)
            batch["gold_span_inp"] = ((batch["labels"] != none_index) & (batch["labels"] != -100)).float()

        elif self.args.gold_span_inp == "label":
            none_index = NerDataset.parse_tag_vocab(self.args.tag_vocab_path).index(self.args.none_tag)
            batch["gold_span_inp"] = torch.where(batch["labels"] != -100, batch["labels"],
                                                 torch.full(batch["labels"].shape, none_index, dtype=torch.int64))

        return batch


def main(args):
    setup_logging()
    parser = HfArgumentParser([AdditionalArguments])
    additional_args = parse_config(parser, args.config)[0]
    set_all_seeds(42)
    dataset = NerDataset(additional_args, corpus_type="test")
    print(len(dataset))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dataset Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
