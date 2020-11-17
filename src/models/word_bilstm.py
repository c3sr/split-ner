import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.base import BaseExecutor
from src.utils.general import set_all_seeds


class WordBiLSTM(nn.Module):

    def __init__(self, word_vocab_size, emb_dim, hidden_dim, out_dim, pre_trained_emb=None):
        super(WordBiLSTM, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if pre_trained_emb != None:
            self.emb = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=False)
        else:
            self.emb = nn.Embedding(self.word_vocab_size, self.emb_dim)
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

    def __init__(self, datapath, vocabpath, tagspath, embpath=None, word_emb_dim=50, none_tag="O", unk_tag="<UNK>",
                 pad_tag="<PAD>", max_seq_len=20):
        super(WordDataset, self).__init__()
        self.datapath = datapath
        self.vocabpath = vocabpath
        self.tagspath = tagspath
        self.embpath = embpath
        self.none_tag = none_tag
        self.unk_tag = unk_tag
        self.pad_tag = pad_tag
        self.max_seq_len = max_seq_len
        self.word_emb = []
        self.word_emb_dim = word_emb_dim
        self.text_sentences = []
        self.word_indexed_sentences = []
        self.word_level_masks = []
        self.tags = []
        self.word_vocab = []
        self.word_vocab_index = dict()
        self.out_tags = []
        self.parse_tags()
        self.parse_vocab()
        self.parse_embfile()
        self.parse_dataset()

    def __len__(self):
        return len(self.word_indexed_sentences)

    def parse_tags(self):
        self.out_tags.append(self.pad_tag)
        with open(self.tagspath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.out_tags.append(line)

    def parse_vocab(self):
        self.word_vocab.append(self.pad_tag)
        with open(self.vocabpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.word_vocab.append(line)
        self.word_vocab.append(self.unk_tag)

        for i, word in enumerate(self.word_vocab):
            self.word_vocab_index[word] = i

    def parse_embfile(self):
        if not self.embpath:
            return
        embdict = dict()
        with open(self.embpath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = line.split(" ")
                    embdict[s[0]] = np.array([float(x) for x in s[1:]], dtype=np.float32)
        embdict[self.unk_tag] = np.zeros(shape=self.word_emb_dim, dtype=np.float32)
        embdict[self.pad_tag] = np.zeros(shape=self.word_emb_dim, dtype=np.float32)
        self.word_emb = np.array([embdict[word] for word in self.word_vocab])

    def parse_dataset(self):
        text_sentences = []
        text_tags = []
        with open(self.datapath, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            text_sentence = []
            text_tag = []
            for row in reader:
                if len(row) == 0:
                    text_sentences.append(text_sentence)
                    text_tags.append(text_tag)
                    text_sentence = []
                    text_tag = []
                else:
                    text_sentence.append(row[0].lower())
                    text_tag.append(row[1])

        for text_sentence, text_tag in zip(text_sentences, text_tags):
            text_sentence, word_indexed_sentence, word_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(
                text_sentence, text_tag)
            self.text_sentences.append(text_sentence)
            self.word_indexed_sentences.append(word_indexed_sentence)
            self.word_level_masks.append(word_level_sentence_mask)
            self.tags.append(indexed_tag)

    def process_sentence_and_tag(self, text_sentence, text_tag):
        if len(text_sentence) > self.max_seq_len:
            word_level_sentence_mask = [1] * self.max_seq_len
            text_sentence = text_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        else:
            word_level_sentence_mask = [0] * (self.max_seq_len - len(text_sentence)) + [1] * len(text_sentence)
            text_sentence = [""] * (self.max_seq_len - len(text_sentence)) + text_sentence
            text_tag = [self.pad_tag] * (self.max_seq_len - len(text_tag)) + text_tag
        word_indexed_sentence = []
        for word in text_sentence:
            if word in self.word_vocab_index:
                word_indexed_sentence.append(self.word_vocab_index[word])
            else:
                word_indexed_sentence.append(self.word_vocab_index[self.unk_tag])
        word_indexed_sentence = np.array(word_indexed_sentence)
        word_level_sentence_mask = np.array(word_level_sentence_mask)
        indexed_tag = np.array([self.out_tags.index(t) for t in text_tag])
        return text_sentence, word_indexed_sentence, word_level_sentence_mask, indexed_tag

    def __getitem__(self, index):
        return self.text_sentences[index], self.word_indexed_sentences[index], self.tags[index]

    def get_query_given_tokens(self, text_sentence):
        text_tag = []
        text_sentence, indexed_sentence, word_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(
            text_sentence, text_tag)
        return text_sentence, indexed_sentence, indexed_tag


class WordBiLSTMExecutor(BaseExecutor):

    def __init__(self, args):
        super(WordBiLSTMExecutor, self).__init__(args)

        self.train_dataset = WordDataset(datapath=self.args.train_path, vocabpath=self.args.word_vocab_path,
                                         tagspath=self.args.tags_path, embpath=self.args.emb_path)
        self.dev_dataset = WordDataset(datapath=self.args.dev_path, vocabpath=self.args.word_vocab_path,
                                       tagspath=self.args.tags_path, embpath=self.args.emb_path)
        self.test_dataset = WordDataset(datapath=self.args.test_path, vocabpath=self.args.word_vocab_path,
                                        tagspath=self.args.tags_path, embpath=self.args.emb_path)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        pre_trained_emb = None
        if not self.args.rand_embedding:
            pre_trained_emb = torch.as_tensor(self.train_dataset.word_emb, device=self.device)
        self.model = WordBiLSTM(word_vocab_size=len(self.train_dataset.word_vocab),
                                emb_dim=self.train_dataset.word_emb_dim,
                                hidden_dim=self.args.hidden_dim, out_dim=len(self.train_dataset.out_tags),
                                pre_trained_emb=pre_trained_emb)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(params=params, lr=self.args.lr)


def main(args):
    set_all_seeds(args.seed)
    executor = WordBiLSTMExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Word-BiLSTM Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="word-bilstm", help="model name (Default: 'word-bilstm')")
    ap.add_argument("--checkpoint_dir", type=str, default="../../checkpoints",
                    help="checkpoints directory (Default: '../../checkpoints')")
    ap.add_argument("--eval", type=str, default="none",
                    help="only evaluate existing checkpoint model (none/best/<checkpoint-id>) (Default: 'none')")
    ap.add_argument("--query", action="store_true",
                    help="query mode, can be used with eval to work with best model (Default: False)")

    ap.add_argument("--data_dir", type=str, default="../../data/GENIA_term_3.02",
                    help="path to input dataset directory (Default: '../../data/GENIA_term_3.02')")
    ap.add_argument("--out_dir", type=str, default="../../data/GENIA_term_3.02/out",
                    help="path to output directory (Default: '../../data/GENIA_term_3.02/out')")
    ap.add_argument("--train_path", type=str, default="train.tsv",
                    help="path to train dataset (Default: 'train.tsv')")
    ap.add_argument("--dev_path", type=str, default="dev.tsv",
                    help="path to dev dataset (Default: 'dev.tsv')")
    ap.add_argument("--test_path", type=str, default="test.tsv",
                    help="path to test dataset (Default: 'test.tsv')")
    ap.add_argument("--word_vocab_path", type=str, default="glove_vocab.txt",
                    help="path to word vocab (Default: 'glove_vocab.txt')")
    ap.add_argument("--tags_path", type=str, default="tag_vocab.txt",
                    help="path to output tags vocab (Default: 'tag_vocab.txt')")
    ap.add_argument("--emb_path", type=str, default="../../../../Embeddings/glove.6B.50d.txt",
                    help="path to pre-trained word embeddings (Default: '../../../../Embeddings/glove.6B.50d.txt')")

    ap.add_argument("--num_epochs", type=int, default=10, help="# epochs to train (Default: 10)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=512, help="LSTM hidden state dimension (Default: 512)")
    ap.add_argument("--rand_embedding", action="store_true",
                    help="randomly initialize word embeddings (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--dropout_ratio", type=float, default=0.5, help="dropout ratio (Default: 0.5)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
