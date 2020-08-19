import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.base import BaseExecutor
from src.utils.char_parsers import OneToOnePatternParser, LowercaseCharParser
from src.utils.general import set_all_seeds


class PatternCNN(nn.Module):
    def __init__(self, inp_dim, conv1_dim, conv2_dim, fc1_dim, out_dim, kernel_size, word_len):
        super(PatternCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inp_dim, out_channels=conv1_dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=conv1_dim, out_channels=conv2_dim, kernel_size=kernel_size)
        fc_inp_dim = (word_len - (kernel_size - 1) * 2) * conv2_dim
        self.fc1 = nn.Linear(in_features=fc_inp_dim, out_features=fc1_dim)
        self.fc2 = nn.Linear(in_features=fc1_dim, out_features=out_dim)

    def forward(self, x):
        batch_size, seq_len, word_len, inp_emb_dim = x.shape
        x = x.view(batch_size * seq_len, word_len, inp_emb_dim)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(batch_size, seq_len, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x


class PatternDataset(Dataset):

    def __init__(self, datapath, tagspath, none_tag="O", max_word_len=20, max_seq_len=20):
        super(PatternDataset, self).__init__()
        self.datapath = datapath
        self.tagspath = tagspath
        self.none_tag = none_tag
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len

        self.special_chars = list(",;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.char_parser = LowercaseCharParser(self.special_chars)
        self.pattern_parser = OneToOnePatternParser(self.special_chars)

        self.text_sentences = []
        self.char_indexed_sentences = []
        self.pattern_indexed_sentences = []
        self.tags = []
        self.char_level_masks = []
        self.out_tags = []
        self.parse_tags()
        self.parse_dataset()

        self.inp_dim = len(self.char_parser.vocab) + len(self.pattern_parser.vocab)

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
            text_sentence = []
            text_tag = []
            for row in reader:
                if len(row) == 0:
                    sentences.append(text_sentence)
                    tags.append(text_tag)
                    text_sentence = []
                    text_tag = []
                else:
                    text_sentence.append(row[0])
                    text_tag.append(row[1])

        if self.none_tag in self.out_tags:
            self.out_tags.remove(self.none_tag)
        self.out_tags = [self.none_tag] + self.out_tags

        for text_sentence, text_tag in zip(sentences, tags):
            text_sentence, char_indexed_sentence, pattern_indexed_sentence, char_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(
                text_sentence, text_tag)
            self.text_sentences.append(text_sentence)
            self.char_indexed_sentences.append(char_indexed_sentence)
            self.pattern_indexed_sentences.append(pattern_indexed_sentence)
            self.char_level_masks.append(char_level_sentence_mask)
            self.tags.append(indexed_tag)

    def process_sentence_and_tag(self, text_sentence, text_tag):
        if len(text_sentence) > self.max_seq_len:
            text_sentence = text_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        elif len(text_sentence) < self.max_seq_len:
            text_sentence += [""] * (self.max_seq_len - len(text_sentence))
            text_tag += [self.none_tag] * (self.max_seq_len - len(text_tag))
        char_indexed_sentence = []
        pattern_indexed_sentence = []
        char_level_sentence_mask = []
        for word in text_sentence:
            char_indexed_word, char_level_word_mask = self.char_parser.get_indexed_text(word)
            pattern_indexed_word, char_level_word_mask = self.pattern_parser.get_indexed_text(word)
            char_indexed_sentence.append(char_indexed_word)
            pattern_indexed_sentence.append(pattern_indexed_word)
            char_level_sentence_mask.append(char_level_word_mask)
        indexed_tag = np.array([self.out_tags.index(t) for t in text_tag])
        char_level_sentence_mask = np.array(char_level_sentence_mask)
        return text_sentence, char_indexed_sentence, pattern_indexed_sentence, char_level_sentence_mask, indexed_tag

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        char_indexed_sentence = []
        for word_index in range(len(self.char_indexed_sentences[index])):
            lowercase_mapping = self.char_parser.get_mapping(
                self.char_indexed_sentences[index][word_index])
            pattern_mapping = self.pattern_parser.get_mapping(self.pattern_indexed_sentences[index][word_index])
            char_indexed_sentence.append(np.hstack([lowercase_mapping, pattern_mapping]))
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        text_sentence = self.text_sentences[index]
        tag = self.tags[index]
        mask = self.char_level_masks[index]
        return text_sentence, char_indexed_sentence, tag

    def get_query_given_tokens(self, text_sentence):
        text_tag = []
        text_sentence, char_indexed_sentence, pattern_indexed_sentence, char_level_sentence_mask, \
        indexed_tag = self.process_sentence_and_tag(text_sentence, text_tag)

        char_indexed_sentence = []
        for word_index in range(len(text_sentence)):
            lowercase_mapping = self.char_parser.get_mapping(char_indexed_sentence[word_index])
            pattern_mapping = self.pattern_parser.get_mapping(pattern_indexed_sentence[word_index])
            char_indexed_sentence.append(np.hstack([lowercase_mapping, pattern_mapping]))
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        return text_sentence, char_indexed_sentence, indexed_tag


class PatternCNNExecutor(BaseExecutor):

    def __init__(self, args):
        super(PatternCNNExecutor, self).__init__(args)

        self.train_dataset = PatternDataset(datapath=self.args.train_path, tagspath=self.args.tags_path)
        self.dev_dataset = PatternDataset(datapath=self.args.dev_path, tagspath=self.args.tags_path)
        self.test_dataset = PatternDataset(datapath=self.args.test_path, tagspath=self.args.tags_path)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size, shuffle=True)

        self.model = PatternCNN(inp_dim=self.train_dataset.inp_dim, conv1_dim=args.conv1_dim, conv2_dim=args.conv2_dim,
                                fc1_dim=args.fc1_dim, out_dim=len(self.train_dataset.out_tags),
                                kernel_size=args.kernel_size, word_len=self.train_dataset.max_word_len)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=args.lr)


def main(args):
    set_all_seeds(args.seed)
    executor = PatternCNNExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pattern-CNN Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="pattern-cnn", help="model name (Default: 'pattern-cnn')")
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

    ap.add_argument("--num_epochs", type=int, default=500, help="# epochs to train (Default: 500)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--conv1_dim", type=int, default=256, help="conv1 layer output channels (Default: 256)")
    ap.add_argument("--conv2_dim", type=int, default=512, help="conv2 layer output channels (Default: 512)")
    ap.add_argument("--fc1_dim", type=int, default=1024, help="fc1 layer output dimension (Default: 1024)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
