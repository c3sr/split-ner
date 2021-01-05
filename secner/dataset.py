import argparse

import torch
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoTokenizer

from secner.additional_args import AdditionalArguments
from secner.utils import Token, set_all_seeds, BertToken, Sentence, parse_config, setup_logging


class NerDataset(Dataset):

    def __init__(self, args, corpus_type):
        super(NerDataset, self).__init__()
        self.args = args
        self.corpus_type = corpus_type
        self.corpus_path = self.set_corpus_path()

        self.tag_vocab = []
        self.parse_tag_vocab()

        self.sentences = []
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        self.bert_start_token, self.bert_end_token = self.get_bert_special_tokens()
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
        self.sentences = NerDataset.read_dataset(self.corpus_path)
        for index in range(len(self.sentences)):
            self.process_sentence(index)

    @staticmethod
    def read_dataset(file_path):
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            tokens = []
            offset = 0
            for line in f:
                line = line.strip()
                if line:
                    row = line.split("\t")
                    if len(row) >= 3:
                        tokens.append(Token(text=row[0], pos_tag=row[1], dep_tag=row[2], tag=row[-1], offset=offset))
                    else:
                        tokens.append(Token(text=row[0], tag=row[-1], offset=offset))
                    offset += 1
                else:
                    sentences.append(Sentence(tokens))
                    tokens = []
                    offset = 0
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        bert_token_ids = [tok.bert_id for tok in sentence.bert_tokens]
        bert_tag_ids = [self.get_tag_index(tok.token.tag) for tok in sentence.bert_tokens]

        return {"input_ids": bert_token_ids, "labels": bert_tag_ids}

    def get_tag_index(self, text_tag):
        if text_tag not in self.tag_vocab:
            text_tag = self.args.none_tag
        return self.tag_vocab.index(text_tag)

    def get_bert_special_tokens(self):
        start_id, end_id = self.tokenizer.encode("")
        start_text, end_text = self.tokenizer.decode([start_id, end_id]).split()
        start_token = BertToken(start_id, Token(start_text, self.args.none_tag, offset=-1))
        end_token = BertToken(end_id, Token(end_text, self.args.none_tag, offset=-1))
        return start_token, end_token

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
                sentence.bert_tokens.append(BertToken(bert_ids[i], bert_token))
        sentence.bert_tokens = sentence.bert_tokens[:self.args.max_seq_len - 1]
        sentence.bert_tokens.append(self.bert_end_token)

    @staticmethod
    def data_collator(features):
        """
        Function not being used in the current flow (can be used later)
        """

        # post-padding
        max_len = max(len(b["labels"]) for b in features)
        # max_len = self.args.max_seq_len
        batch = dict()

        # input_ids
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
        batch["token_type_ids"] = torch.zeros(size=(len(features), max_len), dtype=torch.int64)

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
    ap.add_argument("--config", default="config.json", help="config json file (Default: config.json)")
    ap = ap.parse_args()
    main(ap)
