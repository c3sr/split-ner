import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.base import BaseExecutor
from src.utils.char_parsers import LowercaseCharParser
from src.utils.general import set_all_seeds


class CharCNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, kernel_size, word_len):
        super(CharCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=inp_dim, out_channels=mid_dim, kernel_size=kernel_size)
        fc_inp_dim = (word_len - kernel_size + 1) * mid_dim
        self.fc = nn.Linear(in_features=fc_inp_dim, out_features=out_dim)

    def forward(self, x):
        batch_size, seq_len, word_len, inp_emb_dim = x.shape
        x = x.view(batch_size * seq_len, word_len, inp_emb_dim)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.fc(x)
        return x


class CharDataset(Dataset):

    def __init__(self, datapath, tagspath, none_tag="O", max_word_len=20, max_seq_len=20):
        super(CharDataset, self).__init__()
        self.datapath = datapath
        self.tagspath = tagspath
        self.none_tag = none_tag
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len

        self.special_chars = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.char_parser = LowercaseCharParser(self.special_chars)

        self.text_sentences = []
        self.lowercase_sentences = []
        self.tags = []
        self.masks = []
        self.out_tags = []
        self.parse_tags()
        self.parse_dataset()

        self.inp_dim = len(self.char_parser.vocab)

    def parse_tags(self):
        with open(self.tagspath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

    def parse_dataset(self):
        sentences = []
        tags = []
        with open(self.datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            sentence = []
            tag = []
            for row in reader:
                if len(row) == 0:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
                else:
                    sentence.append(row[0])
                    tag.append(row[1])

        if self.none_tag in self.out_tags:
            self.out_tags.remove(self.none_tag)
        self.out_tags = [self.none_tag] + self.out_tags

        for text_sentence, text_tag in zip(sentences, tags):
            text_sentence, char_indexed_sentence, char_level_mask, indexed_tag = self.process_sentence_and_tag(
                text_sentence, text_tag)
            self.text_sentences.append(text_sentence)
            self.lowercase_sentences.append(char_indexed_sentence)
            self.masks.append(char_level_mask)
            self.tags.append(indexed_tag)

    def process_sentence_and_tag(self, text_sentence, text_tag):
        if len(text_sentence) > self.max_seq_len:
            text_sentence = text_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        elif len(text_sentence) < self.max_seq_len:
            text_sentence += [""] * (self.max_seq_len - len(text_sentence))
            text_tag += [self.none_tag] * (self.max_seq_len - len(text_tag))
        char_indexed_sentence = []
        char_level_mask = []
        for word in text_sentence:
            char_indexed_word, word_mask = self.char_parser.get_indexed_text(word)
            char_indexed_sentence.append(char_indexed_word)
            char_level_mask.append(word_mask)
        char_level_mask = np.array(char_level_mask)
        indexed_tag = np.array([self.out_tags.index(t) for t in text_tag])
        return text_sentence, char_indexed_sentence, char_level_mask, indexed_tag

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        lowercase_sentence = self.lowercase_sentences[index]
        char_indexed_sentence = [self.char_parser.get_mapping(word) for word in lowercase_sentence]
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        mask = self.masks[index]
        return self.text_sentences[index], char_indexed_sentence, self.tags[index]

    def get_query_given_tokens(self, text_sentence):
        text_tag = []
        text_sentence, char_indexed_sentence, char_level_mask, indexed_tag = self.process_sentence_and_tag(
            text_sentence, text_tag)
        char_indexed_sentence = [self.char_parser.get_mapping(word) for word in
                                 char_indexed_sentence]
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        return text_sentence, char_indexed_sentence, text_tag


class CharCNNExecutor(BaseExecutor):

    def __init__(self, args):
        super(CharCNNExecutor, self).__init__(args)

        self.train_dataset = CharDataset(datapath=self.args.train_path, tagspath=self.args.tags_path)
        self.dev_dataset = CharDataset(datapath=self.args.dev_path, tagspath=self.args.tags_path)
        self.test_dataset = CharDataset(datapath=self.args.test_path, tagspath=self.args.tags_path)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size, shuffle=True)

        self.model = CharCNN(inp_dim=self.train_dataset.inp_dim, mid_dim=args.hidden_dim,
                             out_dim=len(self.train_dataset.out_tags),
                             kernel_size=args.kernel_size, word_len=self.train_dataset.max_word_len)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=args.lr)


def main(args):
    set_all_seeds(args.seed)
    executor = CharCNNExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Char-CNN Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="char-cnn", help="model name (Default: 'char-cnn')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoints directory (Default: '../../checkpoints')")
    ap.add_argument("--eval", type=str, default="none",
                    help="only evaluate existing checkpoint model (none/best/<checkpoint-id>) (Default: 'none')")
    ap.add_argument("--query", action="store_true",
                    help="query mode, can be used with eval to work with best model (Default: False)")

    ap.add_argument("--data_dir", type=str, default="../../../GENIA_term_3.02",
                    help="path to input dataset directory (Default: '../../../GENIA_term_3.02')")
    ap.add_argument("--out_dir", type=str, default="../../../GENIA_term_3.02/out",
                    help="path to output directory (Default: '../../../GENIA_term_3.02/out')")
    ap.add_argument("--train_path", type=str, default="train.tsv", help="path to train dataset (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv", help="path to dev dataset (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv", help="path to test dataset (Default: 'test.tsv')")
    ap.add_argument("--tags_path", type=str, default="tag_vocab.txt",
                    help="path to output tags vocab (Default: 'tag_vocab.txt')")

    ap.add_argument("--num_epochs", type=int, default=50, help="# epochs to train (Default: 50)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=256, help="intermediate CNN output channels (Default: 256)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
