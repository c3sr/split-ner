import argparse

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from secner.utils import Token, parse_config, set_all_seeds, BertToken, Sentence, set_absolute_paths


class NerDataset(Dataset):

    def __init__(self, config, corpus_type):
        super(NerDataset, self).__init__()
        self.config = config
        self.corpus_path = self.set_corpus_path(corpus_type)

        self.tag_vocab = []
        self.parse_tag_vocab()

        self.sentences = []
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.bert_start_token, self.bert_end_token = self.get_bert_special_tokens()
        self.parse_dataset()

    def set_corpus_path(self, corpus_type):
        if corpus_type == "train":
            return self.config.data.train_path
        if corpus_type == "dev":
            return self.config.data.dev_path
        if corpus_type == "test":
            return self.config.data.test_path
        return None

    def parse_tag_vocab(self):
        self.tag_vocab.append(self.config.pad_tag)
        with open(self.config.data.tags_path, "r") as f:
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
        text = [tok.text for tok in sentence.tokens]  # original full sentence (with padding), for output file
        bert_token_ids = [tok.bert_id for tok in sentence.bert_tokens]
        offsets = [tok.token.offset for tok in sentence.bert_tokens]
        bert_tag_ids = [self.get_tag_index(tok.token.tag) for tok in sentence.bert_tokens]

        return text, bert_token_ids, offsets, bert_tag_ids

    def get_tag_index(self, text_tag):
        if text_tag not in self.tag_vocab:
            text_tag = self.config.none_tag
        return self.tag_vocab.index(text_tag)

    def get_bert_special_tokens(self):
        start_id, end_id = self.tokenizer.encode("")
        start_text, end_text = self.tokenizer.decode([start_id, end_id]).split()
        start_token = BertToken(start_id, Token(start_text, self.config.none_tag, offset=-1))
        end_token = BertToken(end_id, Token(end_text, self.config.none_tag, offset=-1))
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
        sentence.bert_tokens = sentence.bert_tokens[:self.config.max_seq_len - 1]
        sentence.bert_tokens.append(self.bert_end_token)

    def collate(self, batch):
        # post-padding
        max_len = max(len(b[-1]) for b in batch)
        pad_char = [self.config.pad_tag, 0, -1, 0]
        output = []

        entry = []
        for i in range(len(batch)):
            pad_len = max_len - len(batch[i][0])
            entry.append(batch[i][0] + [pad_char[0]] * pad_len)
        output.append(entry)

        # attention mask
        entry = []
        for i in range(len(batch)):
            good_len = len(batch[i][-1])
            pad_len = max_len - good_len
            entry.append(torch.tensor([1] * good_len + [0] * pad_len))
        output.append(torch.stack(entry))

        for k in range(1, len(batch[0])):
            entry = []
            for i in range(len(batch)):
                pad_len = max_len - len(batch[i][k])
                entry.append(torch.tensor(batch[i][k] + [pad_char[k]] * pad_len))
            output.append(torch.stack(entry))
        return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Dataset Runner")
    ap.add_argument("--config", type=str, default="config.json", help="config json file (Default: config.json)")
    args = ap.parse_args()
    config = parse_config(args.config)
    set_all_seeds(config.seed)
    set_absolute_paths(config)
    dataset = NerDataset(config, corpus_type="test")
    print()
