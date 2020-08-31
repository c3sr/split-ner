import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.base import BaseExecutor
from src.utils.char_parsers import LowercaseCharParser, OneToOnePatternParser, WordCondensedPatternParser, CharParser
from src.utils.evaluator import Evaluator
from src.utils.general import set_all_seeds


class PatternCNN(nn.Module):
    def __init__(self, inp_dim, conv1_dim, hidden_dim, out_dim, kernel_size, word_len, word_vocab_size=None,
                 word_emb_dim=None, pre_trained_emb=None, use_word=True, use_char=True, use_maxpool=False,
                 use_lstm=False):
        super(PatternCNN, self).__init__()

        self.use_maxpool = use_maxpool
        self.use_word = use_word
        self.use_char = use_char
        self.use_lstm = use_lstm
        assert self.use_char or self.use_word, "either of char or word embeddings need to be used"

        if self.use_char:
            print("model using char embeddings")
        if self.use_word:
            print("model using word embeddings")

        next_inp_dim = 0

        if self.use_char:
            # TODO: can try using only single conv. layer as is done generally
            self.conv1 = nn.Conv1d(in_channels=inp_dim, out_channels=conv1_dim, kernel_size=kernel_size)

            if self.use_maxpool:
                next_inp_dim += conv1_dim
            else:
                next_inp_dim += (word_len - (kernel_size - 1) * 1) * conv1_dim

        if self.use_word:
            assert word_vocab_size, "word vocab size needs to be specified"
            assert word_emb_dim, "word embedding dimension needs to be specified"

            next_inp_dim += word_emb_dim

            if isinstance(pre_trained_emb, torch.Tensor):
                self.emb = nn.Embedding.from_pretrained(embeddings=pre_trained_emb, freeze=False)
            else:
                self.emb = nn.Embedding(word_vocab_size, word_emb_dim)

        # TODO: can add a Conv layer (over words) here (instead of LSTM)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=next_inp_dim, hidden_size=hidden_dim // 2, bidirectional=True,
                                batch_first=True)
            next_inp_dim = hidden_dim

        self.fc1 = nn.Linear(in_features=next_inp_dim, out_features=out_dim)

    def forward(self, word_x, char_x, word_mask, char_mask):
        batch_size, seq_len, word_len, inp_emb_dim = char_x.shape

        x = torch.Tensor().to(word_x.device)
        if self.use_char:
            char_x = char_x.view(batch_size * seq_len, word_len, inp_emb_dim)
            char_x = char_x.transpose(1, 2)
            char_x = self.conv1(char_x)
            char_x = F.relu(char_x)

            # TODO: Can try removing ReLU

            if self.use_maxpool:
                char_x, _ = char_x.max(dim=2)
            char_x = char_x.view(batch_size, seq_len, -1)
            x = torch.cat((x, char_x), dim=2)

        if self.use_word:
            word_x = self.emb(word_x)
            x = torch.cat((x, word_x), dim=2)

        if self.use_lstm:
            packed_inp = nn.utils.rnn.pack_padded_sequence(input=x, lengths=word_mask.sum(1).int(), batch_first=True,
                                                           enforce_sorted=False)
            packed_out, _ = self.lstm(packed_inp)
            x, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_out, batch_first=True, total_length=seq_len)

        x = self.fc1(x)

        x = x * word_mask.unsqueeze(-1).expand_as(x)

        return x


class PatternDataset(Dataset):

    def __init__(self, datapath, vocabpath, tagspath, embpath=None, use_char="lower", use_pattern="condensed",
                 include_word_lengths=False, retain_digits=False, pad_tag="<PAD>", unk_tag="<UNK>", word_emb_dim=50,
                 max_word_len=20, max_seq_len=20, post_padding=True):
        super(PatternDataset, self).__init__()
        self.datapath = datapath
        self.vocabpath = vocabpath
        self.tagspath = tagspath
        self.embpath = embpath
        self.use_char = use_char
        self.use_pattern = use_pattern
        self.pad_tag = pad_tag
        self.unk_tag = unk_tag
        self.word_emb_dim = word_emb_dim
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.post_padding = post_padding

        # we prepare char/pattern level embeddings even if not training on it, to get input dimension etc. set
        assert self.use_char != "none" or self.use_pattern != "none", "either char or pattern embeddings need to be used"

        if self.use_char != "none":
            print("dataset using char embeddings")
        if self.use_pattern != "none":
            print("dataset using pattern embeddings")

        if self.use_char == "lower":
            self.char_parser = LowercaseCharParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                   post_padding=post_padding)
        else:
            self.char_parser = CharParser(max_word_len=self.max_word_len, include_special_chars=True,
                                          post_padding=post_padding)

        if self.use_pattern == "one-to-one":
            self.pattern_parser = OneToOnePatternParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                        post_padding=post_padding)
        elif self.use_pattern == "condensed":
            self.pattern_parser = WordCondensedPatternParser(max_word_len=self.max_word_len, include_special_chars=True,
                                                             post_padding=post_padding, retain_digits=retain_digits,
                                                             include_word_lengths=include_word_lengths)

        self.word_vocab = []
        self.word_vocab_index = dict()
        self.word_emb = []
        self.text_sentences = []
        self.word_indexed_sentences = []
        self.char_indexed_sentences = []
        self.pattern_indexed_sentences = []
        self.word_level_masks = []
        self.char_level_masks = []
        self.tags = []
        self.out_tags = []

        self.parse_tags()
        self.parse_vocab()
        self.parse_embfile()
        self.parse_dataset()

        self.inp_dim = 0
        if self.use_char != "none":
            self.inp_dim += len(self.char_parser.vocab)
        if self.use_pattern != "none":
            self.inp_dim += len(self.pattern_parser.vocab)

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
                    embdict[s[0]] = [float(x) for x in s[1:]]
        embdict[self.unk_tag] = [0.0] * self.word_emb_dim
        embdict[self.pad_tag] = [0.0] * self.word_emb_dim
        self.word_emb = np.array([embdict[word] for word in self.word_vocab], dtype=np.float32)

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
                    text_sentence.append(row[0])
                    text_tag.append(row[1])

        for text_sentence, text_tag in zip(text_sentences, text_tags):
            text_sentence, char_indexed_sentence, pattern_indexed_sentence, word_indexed_sentence, \
            word_level_sentence_mask, char_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(
                text_sentence, text_tag)

            self.text_sentences.append(text_sentence)
            self.word_indexed_sentences.append(word_indexed_sentence)
            self.word_level_masks.append(word_level_sentence_mask)
            self.char_indexed_sentences.append(char_indexed_sentence)
            self.pattern_indexed_sentences.append(pattern_indexed_sentence)
            self.char_level_masks.append(char_level_sentence_mask)
            self.tags.append(indexed_tag)

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        char_indexed_sentence = []

        for word_index in range(len(self.text_sentences[index])):
            mappings = []
            if self.use_char != "none":
                indexed_sentence = self.char_indexed_sentences[index][word_index]
                mappings.append(self.char_parser.get_mapping(indexed_sentence))
            if self.use_pattern != "none":
                indexed_sentence = self.pattern_indexed_sentences[index][word_index]
                mappings.append(self.pattern_parser.get_mapping(indexed_sentence))
            char_indexed_sentence.append(np.hstack(mappings))

        text_sentence = self.text_sentences[index]
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        word_indexed_sentence = self.word_indexed_sentences[index]
        word_level_mask = self.word_level_masks[index]
        char_level_mask = self.char_level_masks[index]
        indexed_tag = self.tags[index]
        return text_sentence, word_indexed_sentence, char_indexed_sentence, word_level_mask, char_level_mask, indexed_tag

    def get_query_given_tokens(self, text_sentence):
        text_tag = []
        text_sentence, char_indexed_sentence, pattern_indexed_sentence, word_indexed_sentence, \
        word_level_sentence_mask, char_level_sentence_mask, indexed_tag = self.process_sentence_and_tag(text_sentence,
                                                                                                        text_tag)

        char_indexed_sentence = []
        for word_index in range(len(text_sentence)):
            mappings = []
            if self.use_char != "none":
                mappings.append(self.char_parser.get_mapping(char_indexed_sentence[word_index]))
            if self.use_pattern != "none":
                mappings.append(self.pattern_parser.get_mapping(pattern_indexed_sentence[word_index]))
            char_indexed_sentence.append(np.hstack(mappings))
        char_indexed_sentence = np.array(char_indexed_sentence, dtype=np.float32)
        return text_sentence, word_indexed_sentence, char_indexed_sentence, word_level_sentence_mask, \
               char_level_sentence_mask, indexed_tag

    def process_sentence_and_tag(self, text_sentence, text_tag):
        if len(text_sentence) > self.max_seq_len:
            word_level_sentence_mask = [1] * self.max_seq_len
            text_sentence = text_sentence[:self.max_seq_len]
            text_tag = text_tag[:self.max_seq_len]
        else:
            if self.post_padding:
                word_level_sentence_mask = [1] * len(text_sentence) + [0] * (self.max_seq_len - len(text_sentence))
                text_sentence = text_sentence + [""] * (self.max_seq_len - len(text_sentence))
                text_tag = text_tag + [self.pad_tag] * (self.max_seq_len - len(text_tag))
            else:
                word_level_sentence_mask = [0] * (self.max_seq_len - len(text_sentence)) + [1] * len(text_sentence)
                text_sentence = [""] * (self.max_seq_len - len(text_sentence)) + text_sentence
                text_tag = [self.pad_tag] * (self.max_seq_len - len(text_tag)) + text_tag
        word_indexed_sentence = []
        char_indexed_sentence = []
        pattern_indexed_sentence = []
        char_level_sentence_mask = []
        for index in range(len(text_sentence)):
            word = text_sentence[index]
            if word_level_sentence_mask[index] == 0:
                word_indexed_sentence.append(self.word_vocab_index[self.pad_tag])  # pad tag
            else:
                lw = word.lower()
                word_indexed_sentence.append(self.word_vocab_index[lw] if lw in self.word_vocab_index else
                                             self.word_vocab_index[self.unk_tag])  # unknown tag

            char_indexed_word, char_level_word_mask = self.char_parser.get_indexed_text(word)
            char_level_sentence_mask.append(char_level_word_mask)
            if self.use_char != "none":
                char_indexed_sentence.append(char_indexed_word)
            if self.use_pattern != "none":
                pattern_indexed_word, char_level_word_mask = self.pattern_parser.get_indexed_text(word)
                pattern_indexed_sentence.append(pattern_indexed_word)

        word_indexed_sentence = np.array(word_indexed_sentence)
        char_level_sentence_mask = np.array(char_level_sentence_mask)
        word_level_sentence_mask = np.array(word_level_sentence_mask)
        indexed_tag = np.array([self.out_tags.index(t) for t in text_tag])

        return text_sentence, char_indexed_sentence, pattern_indexed_sentence, word_indexed_sentence, \
               word_level_sentence_mask, char_level_sentence_mask, indexed_tag


class PatternCNNExecutor(BaseExecutor):

    def __init__(self, args):
        super(PatternCNNExecutor, self).__init__(args)

        self.args.word_vocab_path = os.path.join(self.args.data_dir, self.args.word_vocab_path)

        self.use_word = not self.args.no_word
        train_char_emb = self.args.use_char != "none" or self.args.use_pattern != "none"
        post_padding = not self.args.use_pre_padding

        self.train_dataset = PatternDataset(datapath=self.args.train_path, tagspath=self.args.tags_path,
                                            vocabpath=self.args.word_vocab_path, embpath=self.args.emb_path,
                                            unk_tag=self.unk_tag, pad_tag=self.pad_tag, use_char=self.args.use_char,
                                            use_pattern=self.args.use_pattern, word_emb_dim=self.args.word_emb_dim,
                                            max_word_len=self.args.max_word_len, max_seq_len=self.args.max_seq_len,
                                            post_padding=post_padding, retain_digits=self.args.retain_digits,
                                            include_word_lengths=self.args.include_word_lengths)

        # not parsing the embedding file again when processing the dev/test sets

        self.dev_dataset = PatternDataset(datapath=self.args.dev_path, tagspath=self.args.tags_path,
                                          vocabpath=self.args.word_vocab_path, embpath=None, unk_tag=self.unk_tag,
                                          pad_tag=self.pad_tag, use_char=self.args.use_char,
                                          use_pattern=self.args.use_pattern,
                                          word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                          max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                          retain_digits=self.args.retain_digits,
                                          include_word_lengths=self.args.include_word_lengths)

        self.test_dataset = PatternDataset(datapath=self.args.test_path, tagspath=self.args.tags_path,
                                           vocabpath=self.args.word_vocab_path, embpath=None, unk_tag=self.unk_tag,
                                           pad_tag=self.pad_tag, use_char=self.args.use_char,
                                           use_pattern=self.args.use_pattern,
                                           word_emb_dim=self.args.word_emb_dim, max_word_len=self.args.max_word_len,
                                           max_seq_len=self.args.max_seq_len, post_padding=post_padding,
                                           retain_digits=self.args.retain_digits,
                                           include_word_lengths=self.args.include_word_lengths)

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=self.dev_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=args.batch_size, shuffle=True)

        pre_trained_emb = None
        if not self.args.rand_embedding:
            pre_trained_emb = torch.as_tensor(self.train_dataset.word_emb, device=self.device)

        self.model = PatternCNN(inp_dim=self.train_dataset.inp_dim, conv1_dim=self.args.conv1_dim,
                                out_dim=len(self.train_dataset.out_tags), hidden_dim=self.args.hidden_dim,
                                kernel_size=args.kernel_size, word_len=self.train_dataset.max_word_len,
                                word_vocab_size=len(self.train_dataset.word_vocab), use_lstm=self.args.use_lstm,
                                word_emb_dim=self.train_dataset.word_emb_dim, pre_trained_emb=pre_trained_emb,
                                use_char=train_char_emb, use_word=self.use_word, use_maxpool=self.args.use_maxpool)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=args.lr)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_prediction = []
        train_label = []
        with tqdm(self.train_data_loader) as progress_bar:
            for text, word_feature, char_feature, word_mask, char_mask, label in progress_bar:
                word_feature = word_feature.to(self.device)
                char_feature = char_feature.to(self.device)
                word_mask = word_mask.to(self.device)
                char_mask = char_mask.to(self.device)
                label = label.to(self.device)
                batch_size = label.shape[0]
                self.optimizer.zero_grad()
                prediction = self.model(word_feature, char_feature, word_mask, char_mask)
                train_prediction.append(prediction.detach().clone())
                train_label.append(label.detach().clone())
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                progress_bar.set_postfix(Epoch=epoch, Batch_Loss="{0:.3f}".format(loss.item() / batch_size))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        train_prediction = torch.cat(train_prediction, dim=0).cpu().numpy()
        train_prediction = np.argmax(train_prediction, axis=2)
        train_label = torch.cat(train_label, dim=0).cpu().numpy()
        evaluator = Evaluator(gold=train_label, predicted=train_prediction, tags=self.train_dataset.out_tags,
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        print("TRAIN: Epoch: {0} | Loss:{1:.3f} | Token-Level Micro F1: {2:.3f}".format(epoch, train_loss / len(
            self.train_data_loader.dataset), evaluator.significant_token_metric.micro_avg_f1()))

    def evaluate_epoch(self, data_loader, epoch, prefix, outfile=None):
        self.model.eval()
        total_loss = 0.0
        total_text = []
        total_prediction = []
        total_label = []
        for text, word_feature, char_feature, word_mask, char_mask, label in data_loader:
            text = np.array(text).T.tolist()
            word_feature = word_feature.to(self.device)
            char_feature = char_feature.to(self.device)
            word_mask = word_mask.to(self.device)
            char_mask = char_mask.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                prediction = self.model(word_feature, char_feature, word_mask, char_mask)
                total_text.extend(text)
                total_prediction.append(prediction.clone())
                total_label.append(label.clone())
                prediction = prediction.transpose(2, 1)
                loss = self.criterion(prediction, label)
                total_loss += loss.item()
        total_prediction = torch.cat(total_prediction, dim=0).cpu().numpy()
        total_prediction = np.argmax(total_prediction, axis=2)
        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        if outfile:
            self.print_outputs(corpus=total_text, gold=total_label, predicted=total_prediction,
                               mapping=data_loader.dataset.out_tags, outfile=outfile)
        evaluator = Evaluator(gold=total_label, predicted=total_prediction, tags=data_loader.dataset.out_tags,
                              ignore_tags=[self.none_tag, self.pad_tag], none_tag=self.none_tag, pad_tag=self.pad_tag)
        mean_loss = total_loss / len(data_loader.dataset)
        print("{0}: Epoch: {1} | Token-Level Micro F1: {2:.3f} | Loss: {3:.3f}".format(
            prefix, epoch, evaluator.significant_token_metric.micro_avg_f1(), mean_loss))
        print("Entity-Level Metrics:")
        print(evaluator.entity_metric.report())
        print("Token-Level Metrics:")
        print(evaluator.significant_token_metric.report())

        return mean_loss, evaluator

    def query(self, sentence_text):
        self.model.eval()

        sentence_tokens = PatternCNNExecutor.get_query_tokens(sentence_text)
        text, word_feature, char_feature, word_mask, char_mask, _ = self.test_dataset.get_query_given_tokens(
            sentence_tokens)
        word_feature = torch.as_tensor(word_feature, device=self.device).unsqueeze(0)
        char_feature = torch.as_tensor(char_feature, device=self.device).unsqueeze(0)
        word_mask = torch.as_tensor(word_mask, device=self.device).unsqueeze(0)
        char_mask = torch.as_tensor(char_mask, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(word_feature, char_feature, word_mask, char_mask)
        prediction = np.argmax(prediction, axis=2).squeeze(0)
        for i in range(prediction.shape[0]):
            print("{0}\t{1}".format(text[i], self.test_dataset.out_tags[prediction[i]]))


def main(args):
    set_all_seeds(args.seed)
    executor = PatternCNNExecutor(args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pattern-Context-CNN Model for Sequence Labeling")
    ap.add_argument("--name", type=str, default="pattern-context-cnn",
                    help="model name (Default: 'pattern-context-cnn')")
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

    ap.add_argument("--num_epochs", type=int, default=500, help="# epochs to train (Default: 500)")
    ap.add_argument("--batch_size", type=int, default=128, help="batch size (Default: 128)")
    ap.add_argument("--word_emb_dim", type=int, default=50, help="word embedding dimension (Default: 50)")
    ap.add_argument("--max_word_len", type=int, default=30, help="max. #chars in word (Default: 30)")
    ap.add_argument("--max_seq_len", type=int, default=60, help="max. #words in sentence (Default: 60)")
    ap.add_argument("--conv1_dim", type=int, default=128, help="conv1 layer output channels (Default: 128)")
    ap.add_argument("--hidden_dim", type=int, default=512, help="hidden state dim for LSTM, if used (Default: 512)")
    ap.add_argument("--use_maxpool", action="store_true", help="max pool over CNN output to get char embeddings, else "
                                                               "does concatenation (Default: False)")
    ap.add_argument("--kernel_size", type=int, default=5, help="kernel size for CNN (Default: 5)")
    ap.add_argument("--rand_embedding", action="store_true",
                    help="randomly initialize word embeddings (Default: False)")
    ap.add_argument("--use_char", type=str, default="all",
                    help="char embedding type (none/lower/all) (Default: 'all')")
    ap.add_argument("--use_pattern", type=str, default="condensed",
                    help="pattern embedding type (none/one-to-one/condensed) (Default: 'condensed')")
    ap.add_argument("--retain_digits", action="store_true",
                    help="don't replace digits(0-9) with 'd' tag in pattern capturing (Default: False)")
    ap.add_argument("--include_word_lengths", action="store_true",
                    help="include word lengths in pattern capturing (Default: False)")
    ap.add_argument("--use_lstm", action="store_true", help="use LSTM to capture neighbor context (Default: False)")
    ap.add_argument("--no_word", action="store_true", help="don't use word embeddings (Default: False)")
    ap.add_argument("--use_pre_padding", action="store_true", help="pre-padding for char/word (Default: False)")
    ap.add_argument("--lr", type=float, default=0.001, help="learning rate (Default: 0.001)")
    ap.add_argument("--seed", type=int, default=42, help="manual seed for reproducibility (Default: 42)")
    ap.add_argument("--use_cpu", action="store_true", help="force CPU usage (Default: False)")
    ap = ap.parse_args()
    main(ap)
