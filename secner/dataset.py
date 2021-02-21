import argparse
import re
from collections import defaultdict

import torch
from dataclasses import dataclass
from secner.additional_args import AdditionalArguments
from secner.utils.general import Token, set_all_seeds, BertToken, Sentence, parse_config, setup_logging, PairSpan
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer


class NerDataset(Dataset):

    def __init__(self, args: AdditionalArguments, corpus_type):
        super(NerDataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()

        self.tag_vocab = []
        self.parse_tag_vocab()
        self.pos_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.pos_tag_vocab_path, self.args.none_tag)
        self.dep_tag_vocab = NerDataset.parse_aux_tag_vocab(self.args.dep_tag_vocab_path, self.args.none_tag)

        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_mid_sep_token, self.bert_end_token = NerDataset.get_bert_special_tokens(
            self.tokenizer, self.args.none_tag)
        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)
        self.filter_tags()
        self.split_tags()
        self.parse_dataset()

    def set_corpus_path(self):
        if self.corpus_type == "train":
            return self.args.train_path
        if self.corpus_type == "dev":
            return self.args.dev_path
        if self.corpus_type == "test":
            return self.args.test_path
        return None

    def parse_tag_vocab(self):
        with open(self.args.tag_vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.tag_vocab.append(line)

    @staticmethod
    def parse_aux_tag_vocab(vocab_path, none_tag):
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
                        for index in range(sp.start, sp.end + 1):
                            sent.tokens[index].tag = sent.tokens[index].tag[:2] + "Symbolic_simple_chemical"

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
                if tok.tag[2:] not in permissible_tags:
                    tok.tag = self.args.none_tag

    @staticmethod
    def get_spans(sentence):
        spans = defaultdict(list)
        for index, tok in enumerate(sentence.tokens):
            if tok.tag.startswith("B-"):
                spans[tok.tag[2:]].append(PairSpan(index, index))
            elif tok.tag.startswith("I-"):
                spans[tok.tag[2:]][-1].end = index
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
                        tokens.append(Token(text=rt.text, pos_tag=rt.pos_tag, dep_tag=rt.dep_tag, tag=rt.tag,
                                            offset=offset))
                        offset += 1
                else:
                    sentences.append(Sentence(tokens))
                    tokens = []
                    offset = 0
        if args.debug_mode:
            sentences = sentences[:10]
        return sentences

    @staticmethod
    def get_row_tokens(row, args: AdditionalArguments):
        text = row[0]
        tag = row[-1]
        pos_tag = row[1] if len(row) >= 3 else None
        dep_tag = row[2] if len(row) >= 4 else None
        if tag == args.none_tag or args.use_pattern == "none":
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag)]
        pattern_text = NerDataset.make_pattern_type0(text)
        if args.use_pattern == "only":
            return [Token(text=pattern_text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag)]
        if args.use_pattern == "both":
            # E.g.: ABC, a uuu, is investing ...
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag),
                    Token(text=",", pos_tag=",", dep_tag="punct", tag=args.none_tag),
                    Token(text="a", pos_tag="DT", dep_tag="det", tag=args.none_tag),
                    Token(text=pattern_text, pos_tag="NN", dep_tag="appos", tag=args.none_tag),
                    Token(text=",", pos_tag=",", dep_tag="punct", tag=args.none_tag)]

    @staticmethod
    def make_pattern(text, pattern_type):
        if pattern_type == "0":
            return NerDataset.make_pattern_type0(text)
        if pattern_type == "1":
            return NerDataset.make_pattern_type1(text)
        return NotImplementedError

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
        bert_token_type_ids = [tok.token_type for tok in sentence.bert_tokens]
        bert_head_mask = [tok.is_head for tok in sentence.bert_tokens]
        bert_token_text = [tok.token.text for tok in sentence.bert_tokens]
        bert_sub_token_text = [tok.sub_text for tok in sentence.bert_tokens]
        bert_token_pos = [self.pos_tag_vocab.index(tok.token.pos_tag) for tok in sentence.bert_tokens]
        bert_token_dep = [self.dep_tag_vocab.index(tok.token.dep_tag) for tok in sentence.bert_tokens]
        bert_tag_ids = [self.get_tag_index(tok.token.tag) for tok in sentence.bert_tokens]

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
        start_id, end_id = tokenizer.encode("")
        start_text, end_text = tokenizer.decode([start_id, end_id]).split()
        start_token = BertToken(bert_id=start_id,
                                sub_text=start_text,
                                token_type=0,
                                token=Token(start_text, none_tag, offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                                is_head=True)
        mid_sep_token = BertToken(bert_id=end_id,
                                  sub_text=end_text,
                                  token_type=0,
                                  token=Token(end_text, none_tag, offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                                  is_head=True)
        end_token = BertToken(bert_id=end_id,
                              sub_text=end_text,
                              token_type=1,
                              token=Token(end_text, none_tag, offset=-1, pos_tag=none_tag, dep_tag=none_tag),
                              is_head=True)
        return start_token, mid_sep_token, end_token

    def process_sentence(self, index):
        sentence = self.sentences[index]
        sentence.bert_tokens = [self.bert_start_token]
        for token in sentence.tokens:
            out = self.tokenizer(token.text, add_special_tokens=False, return_offsets_mapping=True)
            for i in range(len(out["input_ids"])):
                if i == 0 or not token.tag.startswith("B-"):
                    tag = token.tag
                else:
                    tag = "I-" + token.tag[2:]

                # Handle 'BO' tagging scheme
                if self.args.tagging == "bo" and tag[:2] == "I-":
                    tag = "B-" + tag[2:]

                bert_token = Token(token.text, tag, token.offset, token.pos_tag, token.dep_tag, token.guidance_tag)
                tup = out["offset_mapping"][i]
                sub_text = token.text[tup[0]:tup[1]]
                sentence.bert_tokens.append(BertToken(out["input_ids"][i], sub_text, 0, bert_token, is_head=(i == 0)))
        sentence.bert_tokens = sentence.bert_tokens[:self.args.max_seq_len - 1]
        sentence.bert_tokens.append(self.bert_end_token)

    @staticmethod
    def get_char_ids(batch_text, max_len, vocab):
        max_word_len = max(len(word) for sent in batch_text for word in sent)
        max_word_len = max(max_word_len, 3)
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
    def get_punctuation_vocab_size(punctuation_type):
        if punctuation_type == "type1":
            return 1
        if punctuation_type == "type1-and":
            return 2
        if punctuation_type == "type2":
            return len(list("O.,-/()P"))

    @staticmethod
    def handle_punctuation(word, punctuation_type):
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
        return NotImplementedError

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
        return NotImplementedError

    @staticmethod
    def get_word_type_vocab():
        vocab = list("CSLUFMDPAB")
        return vocab


@dataclass
class NerDataCollator:
    args: AdditionalArguments

    def __call__(self, features):
        # post-padding
        max_len = max(len(entry["labels"]) for entry in features)
        # max_len = self.args.max_seq_len
        batch = dict()

        # input_ids
        if "input_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["input_ids"])
                entry.append(torch.tensor(features[i]["input_ids"] + [0] * pad_len))
            batch["input_ids"] = torch.stack(entry)

        # attention_mask
        entry = []
        for i in range(len(features)):
            good_len = len(features[i]["labels"])
            pad_len = max_len - good_len
            entry.append(torch.tensor([1] * good_len + [0] * pad_len))
        batch["attention_mask"] = torch.stack(entry)

        # token_type_ids
        if "token_type_ids" in features[0]:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["token_type_ids"])
                entry.append(torch.tensor(features[i]["token_type_ids"] + [0] * pad_len))
            batch["token_type_ids"] = torch.stack(entry)

        # char_ids
        if self.args.use_char_cnn in ["char", "both"]:
            batch_text = [entry[self.args.token_type] for entry in features]
            char_vocab = NerDataset.get_char_vocab()
            batch["char_ids"] = NerDataset.get_char_ids(batch_text, max_len, char_vocab)

        # pattern_ids
        if self.args.use_char_cnn in ["pattern", "both"]:
            batch_pattern = [[NerDataset.make_pattern(word, self.args.pattern_type)
                              for word in entry[self.args.token_type]] for entry in features]
            pattern_vocab = NerDataset.get_pattern_vocab(self.args.pattern_type)
            batch["pattern_ids"] = NerDataset.get_char_ids(batch_pattern, max_len, pattern_vocab)

        if self.args.punctuation_handling != "none":
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i][self.args.token_type])
                entry.append(torch.tensor([NerDataset.handle_punctuation(w, self.args.punctuation_handling)
                                           for w in features[i][self.args.token_type]] + [0] * pad_len))
            batch["punctuation_vec"] = torch.stack(entry)

        if self.args.word_type_handling != "none":
            entry = []
            word_type_vocab = NerDataset.get_word_type_vocab()
            for i in range(len(features)):
                pad_len = max_len - len(features[i][self.args.token_type])
                entry.append(torch.tensor([word_type_vocab.index(NerDataset.get_word_type(w))
                                           for w in features[i][self.args.token_type]] + [0] * pad_len))
            batch["word_type_ids"] = torch.stack(entry)

        # head_mask
        entry = []
        for i in range(len(features)):
            pad_len = max_len - len(features[i]["head_mask"])
            entry.append(torch.tensor(features[i]["head_mask"] + [0] * pad_len))
        batch["head_mask"] = torch.stack(entry)

        if self.args.use_pos_tag:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["pos_tag"])
                entry.append(torch.tensor(features[i]["pos_tag"] + [0] * pad_len))
            batch["pos_tag"] = torch.stack(entry)

        if self.args.use_dep_tag:
            entry = []
            for i in range(len(features)):
                pad_len = max_len - len(features[i]["dep_tag"])
                entry.append(torch.tensor(features[i]["dep_tag"] + [0] * pad_len))
            batch["dep_tag"] = torch.stack(entry)

        # labels
        entry = []
        for i in range(len(features)):
            if self.args.use_head_mask:
                labels_mod = [features[i]["labels"][j] for j in range(len(features[i]["labels"])) if
                              features[i]["head_mask"][j]]
            else:
                labels_mod = features[i]["labels"]
            pad_len = max_len - len(labels_mod)
            entry.append(torch.tensor(labels_mod + [-100] * pad_len))
        batch["labels"] = torch.stack(entry)

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
