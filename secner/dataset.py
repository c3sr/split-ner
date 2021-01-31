import argparse

import torch
from dataclasses import dataclass
from secner.additional_args import AdditionalArguments
from secner.utils.general import Token, set_all_seeds, BertToken, Sentence, parse_config, setup_logging
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

        self.sentences = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_mid_sep_token, self.bert_end_token = NerDataset.get_bert_special_tokens(
            self.tokenizer, self.args.none_tag)
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

    def parse_dataset(self):
        self.sentences = NerDataset.read_dataset(self.corpus_path, self.args)
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
        return sentences

    @staticmethod
    def get_row_tokens(row, args: AdditionalArguments):
        text = row[0]
        tag = row[-1]
        pos_tag = row[1] if len(row) >= 3 else None
        dep_tag = row[2] if len(row) >= 4 else None
        if tag == args.none_tag or args.use_pattern == "none":
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag)]
        pattern_text = NerDataset.make_pattern(text)
        if args.use_pattern == "only":
            return [Token(text=pattern_text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag)]
        if args.use_pattern == "both":
            # E.g.: ABC, a uuu, is investing ...
            return [Token(text=text, pos_tag=pos_tag, dep_tag=dep_tag, tag=tag),
                    Token(text=",", tag=args.none_tag),
                    Token(text="a", tag=args.none_tag),
                    Token(text=pattern_text, tag=args.none_tag),
                    Token(text=",", tag=args.none_tag)]

    @staticmethod
    def make_pattern(text):
        pattern_text = ""
        for c in text:
            if "a" <= c <= "z":
                pattern_text += "l"
            elif "A" <= c <= "Z":
                pattern_text += "u"
            else:
                pattern_text += c
        return pattern_text

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        bert_token_ids = [tok.bert_id for tok in sentence.bert_tokens]
        bert_token_type_ids = [tok.token_type for tok in sentence.bert_tokens]
        bert_token_text = [tok.token.text for tok in sentence.bert_tokens]
        bert_tag_ids = [self.get_tag_index(tok.token.tag) for tok in sentence.bert_tokens]

        return {"input_ids": bert_token_ids,
                "token_type_ids": bert_token_type_ids,
                "text": bert_token_text,
                "labels": bert_tag_ids}

    def get_tag_index(self, text_tag):
        if text_tag not in self.tag_vocab:
            text_tag = self.args.none_tag
        return self.tag_vocab.index(text_tag)

    @staticmethod
    def get_bert_special_tokens(tokenizer, none_tag):
        start_id, end_id = tokenizer.encode("")
        start_text, end_text = tokenizer.decode([start_id, end_id]).split()
        start_token = BertToken(start_id, 0, Token(start_text, none_tag, offset=-1))
        mid_sep_token = BertToken(end_id, 0, Token(end_text, none_tag, offset=-1))
        end_token = BertToken(end_id, 1, Token(end_text, none_tag, offset=-1))
        return start_token, mid_sep_token, end_token

    def process_sentence(self, index):
        sentence = self.sentences[index]
        sentence.bert_tokens = [self.bert_start_token]
        for token in sentence.tokens:
            bert_ids = self.tokenizer.encode(token.text, add_special_tokens=False)
            for i in range(len(bert_ids)):
                if i == 0 or not token.tag.startswith("B-"):
                    tag = token.tag
                else:
                    tag = "I-" + token.tag[2:]
                bert_token = Token(token.text, tag, token.offset, token.pos_tag, token.dep_tag, token.guidance_tag)
                sentence.bert_tokens.append(BertToken(bert_ids[i], 0, bert_token))
        sentence.bert_tokens = sentence.bert_tokens[:self.args.max_seq_len - 1]
        sentence.bert_tokens.append(self.bert_end_token)

    @staticmethod
    def get_char_ids(batch_text, max_len, vocab):
        max_word_len = max(len(word) for sent in batch_text for word in sent)
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
    def get_char_vocab():
        # size: 94 (does not include space, newline)
        # additional: can also use list(string.printable) here (size: 100)
        vocab = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        vocab += list("abcdefghijklmnopqrstuvwxyz")
        vocab += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        vocab += list("0123456789")
        return vocab

    @staticmethod
    def get_pattern_vocab():
        vocab = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        vocab += list("ul")
        vocab += list("0123456789")
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
            batch_text = [entry["text"] for entry in features]
            char_vocab = NerDataset.get_char_vocab()
            batch["char_ids"] = NerDataset.get_char_ids(batch_text, max_len, char_vocab)

        # pattern_ids
        if self.args.use_char_cnn in ["pattern", "both"]:
            batch_pattern = [[NerDataset.make_pattern(word) for word in entry["text"]] for entry in features]
            pattern_vocab = NerDataset.get_pattern_vocab()
            batch["pattern_ids"] = NerDataset.get_char_ids(batch_pattern, max_len, pattern_vocab)

        # labels
        entry = []
        for i in range(len(features)):
            pad_len = max_len - len(features[i]["labels"])
            entry.append(torch.tensor(features[i]["labels"] + [-100] * pad_len))
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
