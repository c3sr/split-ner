import argparse
import csv

import numpy as np
import torch.nn as nn
from flair.embeddings import FlairEmbeddings
from torch.utils.data import Dataset

from src.models.base import BaseExecutor
from src.utils.general import set_all_seeds


class WordBiLSTM(nn.Module):

    def __init__(self, word_vocab_size, emb_dim, hidden_dim, out_dim, pre_trained_emb=None):
        super(WordBiLSTM, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.emb = FlairEmbeddings('news-forward')
        # if pre_trained_emb != None:
        #     self.emb = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=False)
        # else:
        #     self.emb = nn.Embedding(self.word_vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 0)
        x, _ = self.lstm(x)
        x = x.transpose(1, 0)
        x = self.fc(x)
        return x


class WordDataset(Dataset):

    def __init__(self, datapath, vocabpath, tagspath, emb_dim=50, none_tag="O", eos_tag="<EOS>",
                 unknown_tag="<UNK>", pad_tag="<PAD>", max_seq_len=20):
        super(WordDataset, self).__init__()
        self.datapath = datapath
        self.vocabpath = vocabpath
        self.tagspath = tagspath
        self.none_tag = none_tag
        self.eos_tag = eos_tag
        self.unknown_tag = unknown_tag
        self.pad_tag = pad_tag
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim  # gets overwritten if embeddings file is provided with different embedding dimensions
        self.sentences = []
        self.tags = []
        self.masks = []
        self.vocab = []
        self.vocabdict = dict()
        self.out_tags = []
        self.parse_tags()
        self.parse_vocab()
        self.parse_dataset()

    def __len__(self):
        return len(self.sentences)

    def parse_tags(self):
        with open(self.tagspath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

    def parse_vocab(self):
        with open(self.vocabpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.vocab.append(line)
        if self.unknown_tag in self.vocab:
            self.vocab.remove(self.unknown_tag)
        if self.pad_tag in self.vocab:
            self.vocab.remove(self.pad_tag)
        self.vocab += [self.unknown_tag, self.pad_tag]

        for i, word in enumerate(self.vocab):
            self.vocabdict[word] = i

    def parse_dataset(self):
        sentences = []
        tags = []
        with open(self.datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            sentence = []
            tag = []
            for row in reader:
                if row[1] == self.eos_tag:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
                else:
                    sentence.append(row[0])
                    tag.append(row[1])
                    self.out_tags.add(row[1])

        if self.none_tag in self.out_tags:
            self.out_tags.remove(self.none_tag)
        if self.pad_tag in self.out_tags:
            self.out_tags.remove(self.pad_tag)
        self.out_tags = [self.none_tag, self.pad_tag] + self.out_tags

        for index in range(len(sentences)):
            if len(sentences[index]) > self.max_seq_len:
                mask = [1.] * self.max_seq_len
                sentences[index] = sentences[index][:self.max_seq_len]
                tags[index] = tags[index][:self.max_seq_len]
            else:
                mask = [1.] * len(sentences[index]) + [0.] * (self.max_seq_len - len(sentences[index]))
                sentences[index] += [""] * (self.max_seq_len - len(sentences[index]))
                tags[index] += [self.pad_tag] * (self.max_seq_len - len(tags[index]))

            self.masks.append(mask)
            self.sentences.append(sentences[index])
            processed_tag = [self.out_tags.index(t) for t in tags[index]]
            self.tags.append(np.array(processed_tag))

    def __getitem__(self, index):
        return self.sentences[index], self.tags[index], self.masks[index]


class WordBiLSTMExecutor(BaseExecutor):

    def __init__(self, args):
        super(WordBiLSTMExecutor, self).__init__(args)

        self.train_dataset = WordDataset(datapath=self.args.train_path, vocabpath=self.args.word_vocab_path,
                                         tagspath=self.args.tags_path, emb_dim=2048)
        # self.dev_dataset = WordDataset(datapath=self.args.dev_path, vocabpath=self.args.word_vocab_path,
        #                                tagspath=self.args.tags_path)
        # self.test_dataset = WordDataset(datapath=self.args.test_path, vocabpath=self.args.word_vocab_path,
        #                                 tagspath=self.args.tags_path)

        # self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        # self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.batch_size, shuffle=True)
        # self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
        #
        # pre_trained_emb = None
        # if not self.args.rand_embedding:
        #     pre_trained_emb = torch.as_tensor(self.train_dataset.emb, device=self.device)
        # self.model = WordBiLSTM(word_vocab_size=len(self.train_dataset.vocab), emb_dim=self.train_dataset.emb_dim,
        #                         hidden_dim=self.args.hidden_dim, out_dim=len(self.train_dataset.out_tags),
        #                         pre_trained_emb=pre_trained_emb)
        #
        # params = filter(lambda p: p.requires_grad, self.model.parameters())
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # self.optimizer = torch.optim.Adam(params=params, lr=self.args.lr)


def main(args):
    set_all_seeds(args.seed)
    executor = WordBiLSTMExecutor(args)
    # executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Word-BiLSTM-Flair Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="word-bilstm-flair", help="model name (Default: 'word-bilstm-flair')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoints directory (Default: '../../checkpoints')")
    ap.add_argument("--eval", type=str, default="none",
                    help="only evaluate existing checkpoint model (none/best/<checkpoint-id>) (Default: 'none')")

    ap.add_argument("--train_path", type=str, default="../../data/GENIA_term_3.02/train.tsv",
                    help="path to train dataset (Default: '../../data/GENIA_term_3.02/train.tsv')")
    ap.add_argument("--dev_path", type=str, default="../../data/GENIA_term_3.02/dev.tsv",
                    help="path to dev dataset (Default: '../../data/GENIA_term_3.02/dev.tsv')")
    ap.add_argument("--test_path", type=str, default="../../data/GENIA_term_3.02/test.tsv",
                    help="path to test dataset (Default: '../../data/GENIA_term_3.02/test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="../../data/GENIA_term_3.02/glove_vocab.txt",
                    help="path to word vocab (Default: '../../data/GENIA_term_3.02/glove_vocab.txt')")
    ap.add_argument("--tags_path", type=str, default="../../data/GENIA_term_3.02/tag_vocab.txt",
                    help="path to output tags vocab (Default: '../../data/GENIA_term_3.02/tag_vocab.txt')")

    ap.add_argument("--num_epochs", type=int, default=10, help="# epochs to train (Default: 10)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=512, help="LSTM hidden state dimension (Default: 512)")
    ap.add_argument("--rand_embedding", action="store_true",
                    help="randomly initialize word embeddings (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
